import math
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, resnet18
from torchvision.models import VGG19_Weights, ResNet18_Weights


# ===========================================================================
# Shared building blocks
# ===========================================================================

def _make_upsample(in_ch: int, scale: int) -> nn.Sequential:
    """
    Two-stage PixelShuffle for scale=4 (2x → 2x), single-stage otherwise.
    Two-stage avoids the checkerboard artifacts that direct 4x PixelShuffle
    reliably produces on high-contrast anime line art.
    """
    if scale == 4:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, 3, 3, 1, 1),
        )
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch * (scale ** 2), 3, 1, 1),
        nn.PixelShuffle(scale),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_ch, 3, 3, 1, 1),
    )


# ---------------------------------------------------------------------------
# Block variants
# ---------------------------------------------------------------------------

class _PlainResBlock(nn.Module):
    """Standard 3×3 conv residual block with learnable scale."""
    def __init__(self, ch: int):
        super().__init__()
        self.body  = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.scale * self.body(x)


class _BSConvBlock(nn.Module):
    """
    Blueprint Separable Convolution residual block.

    BSConv decomposes a standard conv into:
      1. Depthwise 3×3  (spatial mixing, ch×ch params)
      2. Pointwise 1×1  (channel mixing, ch² params)

    This is different from naive depthwise-separable (which does pointwise
    first) — BSConv applies the spatial filter directly on the input features
    without first projecting to a different channel space, which empirically
    preserves edge sharpness better for SR.

    FLOPs: ~(9 + ch) per spatial location vs 18×ch for a plain block.
    For ch=64: ~10x fewer multiply-adds in the body convolutions.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            # Depthwise 3×3
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False),
            # Pointwise 1×1
            nn.Conv2d(ch, ch, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Second BSConv unit
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, 1, 0, bias=True),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.scale * self.body(x)


class _SplitResBlock(nn.Module):
    """
    Channel-split residual block (IMDN/RFDN-style).

    Splits channels in half: one half goes through a cheap depthwise conv,
    the other half skips entirely and is concatenated at the output.
    A final 1×1 mixes channels back together.

    This halves the spatial conv FLOPs compared to a full-channel block while
    maintaining the same receptive field — the skip half acts as a free
    information highway.
    """
    def __init__(self, ch: int):
        super().__init__()
        assert ch % 2 == 0, "ch must be even for SplitResBlock"
        half = ch // 2
        self.process = nn.Sequential(
            nn.Conv2d(half, half, 3, 1, 1, groups=half, bias=False),  # dw
            nn.Conv2d(half, half, 1, 1, 0),                            # pw
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(half, half, 3, 1, 1, groups=half, bias=False),  # dw
            nn.Conv2d(half, half, 1, 1, 0),                            # pw
        )
        self.fuse  = nn.Conv2d(ch, ch, 1, 1, 0)   # mix back after concat
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.half  = half

    def forward(self, x):
        x_proc, x_skip = x[:, :self.half], x[:, self.half:]
        processed = self.process(x_proc)
        fused     = self.fuse(torch.cat([processed, x_skip], dim=1))
        return x + self.scale * fused


# ===========================================================================
# Model variants
# ===========================================================================

class QualityAPISR(nn.Module):
    """
    Quality-first variant. Best output, not real-time.

    128 channels, 16 plain residual blocks.
    Use this to establish your quality ceiling and PSNR target.
    Expected: ~15-25 FPS on a modern GPU at 720p with 4x scale.
    """
    TAG = "quality"

    def __init__(self, scale: int = 4, num_blocks: int = 16, channels: int = 128):
        super().__init__()
        self.scale = scale
        self.head    = nn.Conv2d(3, channels, 3, 1, 1)
        self.body    = nn.Sequential(*[_PlainResBlock(channels) for _ in range(num_blocks)])
        self.upsample = _make_upsample(channels, scale)

    def forward(self, x):
        f = self.head(x)
        return torch.clamp(self.upsample(f + self.body(f)), 0.0, 1.0)


class MidAPISR(nn.Module):
    """
    Middle-ground variant. Plain convs (not depthwise), narrower than Quality.

    96 channels, 14 plain residual blocks (~2.99M params — roughly half of
    QualityAPISR's 5.91M, vs BSConv's 0.41M / FastAPISR's 0.20M).

    This exists because BSConv's depthwise convs are memory-bound rather
    than compute-bound: shrinking params doesn't translate proportionally
    into speed, and the quality hit is larger than the param reduction
    would suggest. A narrower *plain*-conv model avoids that issue —
    every parameter reduction here translates roughly linearly into both
    fewer FLOPs and less VRAM traffic, with no depthwise penalty.

    Expected quality: noticeably closer to QualityAPISR than the smaller
    1.6M config, since width (channel count) is the bigger quality lever
    for SR feature richness — this keeps width nearly unchanged from
    Quality (96 vs 128) and only trims depth (14 vs 16 blocks).

    Expected: at half the params of Quality (which hit ~22 FPS with TRT
    FP16), this should comfortably clear 30 FPS.

    Other configs at nearby param counts, if you want to tune further:
      channels=112, num_blocks=10  → 3.17M params (wider, shallower)
      channels=80,  num_blocks=10  → 1.62M params (smaller/faster fallback)
    Pass them directly: MidAPISR(scale=4, channels=112, num_blocks=10)
    """
    TAG = "mid"

    def __init__(self, scale: int = 4, num_blocks: int = 14, channels: int = 96):
        super().__init__()
        self.scale = scale
        self.head     = nn.Conv2d(3, channels, 3, 1, 1)
        self.body     = nn.Sequential(*[_PlainResBlock(channels) for _ in range(num_blocks)])
        self.upsample = _make_upsample(channels, scale)

    def forward(self, x):
        f = self.head(x)
        return torch.clamp(self.upsample(f + self.body(f)), 0.0, 1.0)


class BSConvAPISR(nn.Module):
    """
    Blueprint Separable Conv variant. Best quality/speed tradeoff.

    64 channels, 12 BSConv residual blocks.
    The BSConv body is ~10x cheaper per block than QualityAPISR's plain
    convs, with a modest quality penalty (typically 0.3-0.5 dB PSNR).
    For anime this penalty is smaller than for photorealistic SR because
    the content is already stylised — perceptual quality holds up well.
    Expected: ~50-80 FPS on a modern GPU at 720p with 4x scale.
    """
    TAG = "bsconv"

    def __init__(self, scale: int = 4, num_blocks: int = 12, channels: int = 64):
        super().__init__()
        self.scale = scale
        self.head     = nn.Conv2d(3, channels, 3, 1, 1)
        self.body     = nn.Sequential(*[_BSConvBlock(channels) for _ in range(num_blocks)])
        self.upsample = _make_upsample(channels, scale)

    def forward(self, x):
        f = self.head(x)
        return torch.clamp(self.upsample(f + self.body(f)), 0.0, 1.0)


class FastAPISR(nn.Module):
    """
    Real-time variant. Designed to hit 60+ FPS at 1080p input (4x → 4K).

    48 channels, 8 channel-split residual blocks.
    The split block halves the body's spatial conv FLOPs relative to BSConv
    while the channel count reduction further cuts the upsample cost.
    Quality drop vs BSConvAPISR is ~0.5-1.0 dB PSNR — perceptually still
    clean on anime given its flat regions and sharp edges.
    Expected: ~120-200 FPS on a modern GPU at 720p with 4x scale.
    With TensorRT FP16: likely 2-3x on top of that.
    """
    TAG = "fast"

    def __init__(self, scale: int = 4, num_blocks: int = 8, channels: int = 48):
        super().__init__()
        self.scale = scale
        self.head     = nn.Conv2d(3, channels, 3, 1, 1)
        self.body     = nn.Sequential(*[_SplitResBlock(channels) for _ in range(num_blocks)])
        self.upsample = _make_upsample(channels, scale)

    def forward(self, x):
        f = self.head(x)
        return torch.clamp(self.upsample(f + self.body(f)), 0.0, 1.0)


# Alias so train.py import (from apisr import SimpleAPISR) still works
# Point it at BSConvAPISR as the default training target going forward
SimpleAPISR = BSConvAPISR

# Registry for train.py --model flag
MODEL_REGISTRY: dict[str, type] = {
    "quality": QualityAPISR,
    "mid":     MidAPISR,
    "bsconv":  BSConvAPISR,
    "fast":    FastAPISR,
}


# ===========================================================================
# Loss
# ===========================================================================

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(x.device)
    std  = _IMAGENET_STD.to(x.device)
    return (x - mean) / std


class _VGGFeatureExtractor(nn.Module):
    """VGG19 relu3_4 — photorealistic texture features."""
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:23].eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.net = vgg

    def forward(self, x):
        return self.net(_imagenet_norm(x))


class _ResNetFeatureExtractor(nn.Module):
    """
    ResNet18 layer2 + layer3 — anime twin proxy.
    Captures mid-level line/shape features that VGG misses.
    """
    def __init__(self, device):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(device)
        for p in r.parameters():
            p.requires_grad = False
        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3

    def forward(self, x):
        x  = self.layer1(self.stem(_imagenet_norm(x)))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


class _FrequencyLoss(nn.Module):
    """
    FFT magnitude L1 loss.
    Directly penalises missing high-frequency content — critical for anime
    line art that pixel / perceptual losses tend to blur.
    """
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        def grey(t):
            return 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]
        return F.l1_loss(
            torch.abs(torch.fft.rfft2(grey(sr), norm='ortho')),
            torch.abs(torch.fft.rfft2(grey(hr), norm='ortho')),
        )


class TwinPerceptualLoss(nn.Module):
    """
    Anime-tuned Twin Perceptual Loss with staged warmup.

    Training in two phases prevents the 19 dB collapse that happens when
    perceptual losses are active from epoch 1:

    Phase 1 — pixel warmup (epochs 0..warmup_epochs):
        Loss = L1 only.
        The model learns basic structure and colour before perceptual
        gradients (which are noisy at random init) pull it off course.

    Phase 2 — full loss (epochs warmup_epochs..end):
        Loss = L1 + VGG + ResNet + FFT, introduced gradually via a
        ramp factor that goes from 0 → 1 over `ramp_epochs` epochs.
        This avoids a sudden loss spike at the phase boundary.

    Term              Weight   Purpose
    ──────────────────────────────────────────────────────────────────
    L1 pixel          1.0      Fidelity baseline, prevents hallucination
    VGG relu3_4       0.01     Photorealistic texture & colour coherence
    ResNet layer2     0.005    Mid-level shape (anime twin proxy)
    ResNet layer3     0.005    Higher-level structure
    FFT magnitude     0.01     Line-art sharpness
    ──────────────────────────────────────────────────────────────────
    """
    def __init__(
        self,
        device:        torch.device,
        warmup_epochs: int = 20,
        ramp_epochs:   int = 10,
    ):
        super().__init__()
        self.vgg    = _VGGFeatureExtractor(device)
        self.resnet = _ResNetFeatureExtractor(device)
        self.freq   = _FrequencyLoss()
        self.l1     = nn.L1Loss()
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs   = ramp_epochs
        self._epoch        = 0        # updated by set_epoch() each epoch

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch so the loss knows its phase."""
        self._epoch = epoch

    def _perceptual_weight(self) -> float:
        """
        0.0 during warmup, ramps linearly 0→1 over ramp_epochs,
        then stays at 1.0.
        """
        e = self._epoch
        if e < self.warmup_epochs:
            return 0.0
        ramp = (e - self.warmup_epochs) / max(self.ramp_epochs, 1)
        return min(ramp, 1.0)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        loss = self.l1(sr, hr)
        w = self._perceptual_weight()
        if w > 0.0:
            loss = loss + w * 0.01  * self.l1(self.vgg(sr), self.vgg(hr))
            sr_f2, sr_f3 = self.resnet(sr)
            hr_f2, hr_f3 = self.resnet(hr)
            loss = loss + w * 0.005 * self.l1(sr_f2, hr_f2)
            loss = loss + w * 0.005 * self.l1(sr_f3, hr_f3)
            loss = loss + w * 0.01  * self.freq(sr, hr)
        return loss


# ===========================================================================
# Benchmark
# ===========================================================================

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_gflops(model: nn.Module, tile_hw: int) -> float:
    """Rough GFLOPs estimate via fvcore if available, else returns -1."""
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.zeros(1, 3, tile_hw, tile_hw)
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        return flops.total() / 1e9
    except Exception:
        return -1.0


def benchmark(
    tile_hw:    int  = 64,     # LR tile size fed to the model
    warmup:     int  = 20,
    iterations: int  = 200,
    device_str: str  = "cuda",
) -> None:
    """
    Benchmark all three variants on synthetic tiles.

    tile_hw is the LR input size. For 4x scale, output will be tile_hw*4.
    A tile of 64 LR px → 256 HR px is a reasonable real-time tile size.

    Run this on your actual GPU before choosing a model to train:
        python apisr.py --benchmark
        python apisr.py --benchmark --tile 128   # larger tiles
        python apisr.py --benchmark --device cpu  # CPU fallback check
    """
    device  = torch.device(device_str if torch.cuda.is_available() else "cpu")
    scale   = 4
    x       = torch.randn(1, 3, tile_hw, tile_hw, device=device)

    variants = [
        QualityAPISR(scale=scale),
        MidAPISR(scale=scale),
        BSConvAPISR(scale=scale),
        FastAPISR(scale=scale),
    ]

    print(f"\n{'─'*65}")
    print(f"  Benchmark — device: {device}  |  LR tile: {tile_hw}×{tile_hw}"
          f"  →  HR: {tile_hw*scale}×{tile_hw*scale}")
    print(f"{'─'*65}")
    print(f"  {'Model':<18} {'Params':>8}  {'GFLOPs':>8}  "
          f"{'ms/tile':>8}  {'FPS (1 tile)':>13}  {'FPS (batch-8)':>14}")
    print(f"{'─'*65}")

    for model in variants:
        model = model.to(device).eval()
        gflops = _estimate_gflops(model, tile_hw)

        # Single-tile throughput
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        ms_single = (t1 - t0) / iterations * 1000
        fps_single = 1000 / ms_single

        # Batch-8 throughput (simulates 8 tiles in one forward pass)
        x8 = torch.randn(8, 3, tile_hw, tile_hw, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x8)
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = model(x8)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        ms_batch   = (t1 - t0) / iterations * 1000
        fps_batch8 = 8 * 1000 / ms_batch   # tiles per second

        params_m = _count_params(model) / 1e6
        gflops_s = f"{gflops:.2f}" if gflops > 0 else "  n/a"

        print(
            f"  {type(model).__name__:<18} "
            f"{params_m:>6.2f}M  "
            f"{gflops_s:>8}  "
            f"{ms_single:>8.2f}  "
            f"{fps_single:>13.1f}  "
            f"{fps_batch8:>14.1f}"
        )

    print(f"{'─'*65}")
    print(
        "  FPS (batch-8) = throughput when 8 tiles from the same frame are\n"
        "  batched into one forward pass — this is the real-time relevant\n"
        "  number since the C++ tiler will batch tiles per frame.\n"
        "  Target for real-time 24fps video: FPS (batch-8) > ~200\n"
        "  (accounts for decode + stitch overhead on top of inference)\n"
    )


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark across all variants")
    parser.add_argument("--tile",   type=int, default=64,
                        help="LR tile size for benchmark (default 64)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(tile_hw=args.tile, device_str=args.device)
    else:
        parser.print_help()
