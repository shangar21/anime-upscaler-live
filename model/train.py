import argparse
import logging
import math
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from dataset import SRDataset
from apisr import MODEL_REGISTRY, TwinPerceptualLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR between two [0, 1] tensors."""
    mse = torch.mean((img1 - img2) ** 2).item()
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


# ---------------------------------------------------------------------------
# Cosine blend mask (mirrors the C++ tiler so Python eval is comparable)
# ---------------------------------------------------------------------------

def _make_blend_mask(size: int, overlap: int) -> np.ndarray:
    """
    Cosine weight mask: 1.0 in the centre, tapers to 0 at edges within
    `overlap` pixels. Matches the C++ cosine blend in tiling.hpp.
    """
    mask = np.ones((size, size), dtype=np.float32)
    for i in range(overlap):
        w = 0.5 * (1.0 - math.cos(math.pi * i / overlap))
        mask[i,          :] = np.minimum(mask[i,          :], w)
        mask[size - 1 - i, :] = np.minimum(mask[size - 1 - i, :], w)
        mask[:,          i] = np.minimum(mask[:,          i], w)
        mask[:, size - 1 - i] = np.minimum(mask[:, size - 1 - i], w)
    return mask  # (H, W)


# ---------------------------------------------------------------------------
# Mid-training visual evaluation (PyTorch model)
# ---------------------------------------------------------------------------

def evaluate(
    model:     torch.nn.Module,
    data_dir:  str,
    epoch:     int,
    device:    torch.device,
    sr_rate:   int,
    num_samples: int = 3,
) -> None:
    """
    Save a few SR samples during training so you can eyeball quality.
    Uses the model's own scale factor (sr_rate), not 2*sr_rate.
    """
    model.eval()
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ][:num_samples]

    with torch.no_grad():
        for i, filename in enumerate(files):
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            w, h = img.size
            # Crop to a multiple of sr_rate so LR→HR is lossless
            img = img.crop((0, 0, w - w % sr_rate, h - h % sr_rate))
            lr_img = img.resize(
                (img.width // sr_rate, img.height // sr_rate), Image.LANCZOS
            )
            tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)
            out    = model(tensor)
            save_image(out, os.path.join(out_dir, f"epoch_{epoch+1}_sample_{i}.png"))

    model.train()


# ---------------------------------------------------------------------------
# Final patch-based evaluation (PyTorch model, with overlap-blend stitching)
# ---------------------------------------------------------------------------

def final_eval(
    model:    torch.nn.Module,
    data_dir: str,
    device:   torch.device,
    hr_size:  int,
    sr_rate:  int,
    overlap:  int = 32,
) -> None:
    """
    Evaluate with proper overlapping tile stitching.

    The previous version had tiles on a non-overlapping grid — the blend mask
    was computed but neighbouring tiles never overlapped, so it did nothing and
    tile boundaries showed as grid lines.

    Fix: step in HR space = hr_size - overlap, so adjacent tiles share
    `overlap` pixels on each edge and the cosine mask actually blends them.
    """
    model.eval()
    files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    lr_size  = hr_size // sr_rate
    lr_overlap = overlap // sr_rate          # overlap in LR input space
    lr_step    = lr_size - 2 * lr_overlap    # step in LR space
    hr_step    = lr_step * sr_rate           # corresponding step in HR space
    assert lr_step > 0, "overlap must be < lr_size // 2"

    # Blend mask lives in HR output space — size matches model output tile
    mask_hr = torch.from_numpy(
        _make_blend_mask(hr_size, overlap)
    ).to(device)                             # (hr_size, hr_size)

    total_psnr = 0.0
    total_time = 0.0
    n_evaluated = 0

    logger.info("Final evaluation on %d images", len(files))

    with torch.no_grad():
        for filename in tqdm(files, desc="eval"):
            hr_img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            W, H   = hr_img.size
            if W < hr_size or H < hr_size:
                continue

            # Crop so the overlapping tile grid covers the whole image cleanly
            # ceil_div so we don't lose the right/bottom edge
            nw = math.ceil((W - overlap) / hr_step)
            nh = math.ceil((H - overlap) / hr_step)
            cW = nw * hr_step + overlap
            cH = nh * hr_step + overlap
            # Pad with reflection if crop would exceed image bounds
            pad_r = max(0, cW - W)
            pad_b = max(0, cH - H)
            if pad_r > 0 or pad_b > 0:
                hr_img = TF.pad(hr_img, [0, 0, pad_r, pad_b], padding_mode='reflect')
            hr_img_c  = hr_img.crop((0, 0, cW, cH))
            hr_tensor = TF.to_tensor(hr_img_c).to(device)

            oH, oW    = cH, cW   # output is same spatial size (model upscales LR)
            # Wait — we need to upsample coords: LR → HR
            # Each LR patch of lr_size → HR patch of hr_size
            # The full output canvas is cH * sr_rate × cW * sr_rate... but we
            # compare against the original HR so keep at HR resolution directly.
            sr_accum   = torch.zeros(3, cH, cW, device=device)
            weight_acc = torch.zeros(1, cH, cW, device=device)

            t0 = time.perf_counter()

            for row in range(nh + 1):
                for col in range(nw + 1):
                    # HR-space tile origin (clamped so tile fits within canvas)
                    hy = min(row * hr_step, cH - hr_size)
                    hx = min(col * hr_step, cW - hr_size)

                    # Extract HR patch, downsample to LR, run SR
                    hr_patch = hr_img_c.crop((hx, hy, hx + hr_size, hy + hr_size))
                    lr_patch = hr_patch.resize((lr_size, lr_size), Image.LANCZOS)
                    lr_t     = TF.to_tensor(lr_patch).unsqueeze(0).to(device)
                    sr_patch = model(lr_t).squeeze(0)   # (3, hr_size, hr_size)

                    sr_accum [:, hy:hy+hr_size, hx:hx+hr_size] += sr_patch * mask_hr
                    weight_acc[:, hy:hy+hr_size, hx:hx+hr_size] += mask_hr

            sr_full = (sr_accum / weight_acc.clamp(min=1e-6)).clamp(0, 1)
            # Trim padding before PSNR comparison
            sr_full   = sr_full[:, :H, :W]
            hr_tensor = TF.to_tensor(
                Image.open(os.path.join(data_dir, filename)).convert('RGB')
            ).to(device)[:, :H, :W]

            total_time += time.perf_counter() - t0
            total_psnr += calculate_psnr(hr_tensor, sr_full)
            n_evaluated += 1

    if n_evaluated == 0:
        logger.warning("No images were large enough to evaluate.")
        return

    logger.info("Avg PSNR : %.2f dB", total_psnr / n_evaluated)
    logger.info("Avg speed: %.2f FPS", n_evaluated / total_time)
    save_image(sr_full, "final_reconstruction_sample.png")
    model.train()


# ---------------------------------------------------------------------------
# ONNX evaluation (matches C++ pipeline behaviour)
# ---------------------------------------------------------------------------

def evaluate_onnx(
    onnx_path:   str,
    data_dir:    str,
    sr_rate:     int,
    hr_size:     int,
    device_type: str = 'cuda',
    overlap:     int = 32,
) -> None:
    """
    Tile-based ONNX inference with overlapping cosine blend stitching.
    device_type: 'cuda' | 'amd' | 'cpu'
    """
    provider_map = {
        'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'amd':  ['ROCMExecutionProvider', 'CPUExecutionProvider'],
        'cpu':  ['CPUExecutionProvider'],
    }
    providers = provider_map.get(device_type, ['CPUExecutionProvider'])

    session    = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    logger.info("ONNX providers in use: %s", session.get_providers())

    out_dir = "eval_results_onnx"
    os.makedirs(out_dir, exist_ok=True)

    lr_size    = hr_size // sr_rate
    lr_overlap = overlap // sr_rate
    lr_step    = lr_size - 2 * lr_overlap
    hr_step    = lr_step * sr_rate
    assert lr_step > 0, "overlap must be < lr_size // 2"
    mask = _make_blend_mask(hr_size, overlap)

    files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not files:
        logger.warning("No images found in %s", data_dir)
        return

    test_file = files[0]
    hr_img    = Image.open(os.path.join(data_dir, test_file)).convert('RGB')
    W, H      = hr_img.size

    nw = math.ceil((W - overlap) / hr_step)
    nh = math.ceil((H - overlap) / hr_step)
    cW = nw * hr_step + overlap
    cH = nh * hr_step + overlap
    if cW > W or cH > H:
        from PIL import ImageOps
        hr_img = ImageOps.expand(hr_img, (0, 0, max(0, cW - W), max(0, cH - H)))
    hr_img = hr_img.crop((0, 0, cW, cH))

    sr_accum  = np.zeros((3, cH, cW), dtype=np.float32)
    weight_ac = np.zeros((1, cH, cW), dtype=np.float32)

    t0 = time.perf_counter()
    n_tiles = 0

    for row in range(nh + 1):
        for col in range(nw + 1):
            hy = min(row * hr_step, cH - hr_size)
            hx = min(col * hr_step, cW - hr_size)

            hr_patch = hr_img.crop((hx, hy, hx + hr_size, hy + hr_size))
            lr_patch = hr_patch.resize((lr_size, lr_size), Image.LANCZOS)
            lr_np    = (
                np.array(lr_patch, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )[np.newaxis]

            out_patch = session.run(None, {input_name: lr_np})[0].squeeze(0)
            sr_accum [:, hy:hy+hr_size, hx:hx+hr_size] += out_patch * mask
            weight_ac[:, hy:hy+hr_size, hx:hx+hr_size] += mask
            n_tiles += 1

    elapsed = time.perf_counter() - t0
    sr_full = np.clip(sr_accum / np.maximum(weight_ac, 1e-6), 0, 1)
    sr_full = sr_full[:, :H, :W]   # trim padding

    save_image(
        torch.from_numpy(sr_full),
        os.path.join(out_dir, "final_stitched_onnx.png"),
    )
    logger.info("Processed %d tiles in %.3fs (%.2f FPS)",
                n_tiles, elapsed, 1.0 / elapsed)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model_path: str,
    onnx_path:  str,
    sr_rate:    int = 4,
    hr_size:    int = 192,
    model_tag:  str = "bsconv",
) -> None:
    """
    Load a saved .pth and export to ONNX with dynamic spatial axes so the
    C++ tiler can feed any tile size at runtime.
    """
    device    = torch.device("cpu")   # CPU export is safest for cross-platform compat
    model_cls = MODEL_REGISTRY[model_tag]
    model     = model_cls(scale=sr_rate)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    lr_size     = hr_size // sr_rate
    dummy_input = torch.randn(1, 3, lr_size, lr_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
    )
    logger.info("Exported model → %s", onnx_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    parser = argparse.ArgumentParser(description="Train anime APISR")
    parser.add_argument("--data_dir",   type=str,   required=True)
    parser.add_argument("--sr_rate",    type=int,   default=4)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes (default: 8). "
                             "Increase if GPU utilization is low — small "
                             "models like bsconv/fast often starve the GPU "
                             "with too few workers.")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="Batches each worker prefetches ahead (default: 4)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--hr_size",    type=int,   default=192)
    parser.add_argument("--model", type=str, default="mid",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model variant: quality | mid | bsconv | fast "
                             "(default: mid). 'mid' is plain convs at ~1.6M "
                             "params — best quality/speed tradeoff if "
                             "bsconv's depthwise blocks hurt quality too much.")
    parser.add_argument("--downsample", type=str,   default='lanczos',
                        choices=['bicubic', 'lanczos'])
    parser.add_argument("--save_path",  type=str,   default="apisr.pth")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                        help="Epochs of L1-only training before perceptual "
                             "loss is introduced (default: 20)")
    parser.add_argument("--ramp_epochs",   type=int, default=10,
                        help="Epochs to ramp perceptual loss 0→full weight "
                             "after warmup ends (default: 10)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    # --- Dataset ---
    dataset = SRDataset(
        root_dir=args.data_dir,
        hr_size=args.hr_size,
        sr_rate=args.sr_rate,
        downsample=args.downsample,
        color_jitter=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )
    logger.info(
        "DataLoader: batch_size=%d  num_workers=%d  prefetch_factor=%d",
        args.batch_size, args.num_workers, args.prefetch_factor,
    )

    # --- Model ---
    model_cls = MODEL_REGISTRY[args.model]
    model     = model_cls(scale=args.sr_rate).to(device)
    logger.info("Model variant : %s  (%s params)",
                args.model,
                f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        # Infer epoch from filename if possible (e.g. "apisr_epoch42.pth")
        try:
            start_epoch = int(
                os.path.splitext(args.resume)[0].split("epoch")[-1]
            )
        except ValueError:
            pass
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

    # --- Loss & optimiser ---
    criterion = TwinPerceptualLoss(
        device,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
    )
    logger.info(
        "Loss schedule: L1-only for %d epochs, then ramp perceptual "
        "loss over %d epochs", args.warmup_epochs, args.ramp_epochs
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # Cosine annealing with warm restarts: much better than StepLR for SR.
    # T_0=30 means the LR resets every 30 epochs; T_mult=2 doubles the
    # period each restart so later restarts are longer exploratory phases.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        criterion.set_epoch(epoch)
        pw = criterion._perceptual_weight()
        phase = (
            "L1-only warmup" if pw == 0.0
            else f"perceptual ramp ({pw:.2f})" if pw < 1.0
            else "full loss"
        )
        model.train()
        epoch_loss   = 0.0
        data_time    = 0.0   # time spent waiting for the dataloader
        compute_time = 0.0   # time spent in forward+backward+step
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}] [{phase}]", leave=True)

        t_prev = time.perf_counter()
        for lr_imgs, hr_imgs in loop:
            t_data = time.perf_counter()
            data_time += t_data - t_prev   # time since last iteration ended

            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            outputs = model(lr_imgs)
            loss    = criterion(outputs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clip — prevents the ResNet/VGG perceptual gradients
            # from occasionally spiking early in training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()   # accurate compute timing
            t_compute = time.perf_counter()
            compute_time += t_compute - t_data
            t_prev = t_compute

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        data_pct = 100 * data_time / (data_time + compute_time + 1e-9)
        logger.info(
            "Epoch %d  avg_loss=%.4f  data_wait=%.1fs (%.0f%%)  compute=%.1fs",
            epoch + 1, avg_loss, data_time, data_pct, compute_time,
        )
        if data_pct > 30:
            logger.warning(
                "GPU spent %.0f%% of this epoch waiting for data — "
                "increase --num_workers or --prefetch_factor. "
                "If data_wait stays high even after that, your CPU "
                "(PIL crop/resize/jitter) is the bottleneck, not the GPU.",
                data_pct,
            )

        # Visual eval every 5 epochs
        if (epoch + 1) % 5 == 0:
            evaluate(model, args.data_dir, epoch, device, args.sr_rate)

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = f"{os.path.splitext(args.save_path)[0]}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt)
            logger.info("Checkpoint saved → %s", ckpt)

    # --- Final save & export ---
    torch.save(model.state_dict(), args.save_path)
    logger.info("Model saved → %s", args.save_path)

    onnx_path = os.path.splitext(args.save_path)[0] + '.onnx'
    export_to_onnx(
        args.save_path, onnx_path,
        sr_rate=args.sr_rate,
        hr_size=args.hr_size,
        model_tag=args.model,
    )

    logger.info("Running final evaluation...")
    final_eval(model, args.data_dir, device, args.hr_size, args.sr_rate)
    evaluate_onnx(onnx_path, args.data_dir, args.sr_rate, args.hr_size)


if __name__ == "__main__":
    train()
