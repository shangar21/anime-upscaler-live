"""
quant.py — TensorRT export and inference benchmark

Workflow:
    # 1. Export TRT engines from a trained ONNX
    python quant.py export --onnx model.onnx --out_dir trt_engines/

    # 2. Benchmark all backends on your data
    python quant.py bench --onnx model.onnx --trt_dir trt_engines/ \\
                          --data_dir /your/anime/images --sr_rate 4 --hr_size 192

Outputs a table like:
    Backend          FPS (batch-8)    PSNR (dB)    Notes
    ORT FP32         48.2             32.41        baseline
    TRT FP32         91.7             32.41        ~1.9x
    TRT FP16         187.3            32.38        ~3.9x, -0.03 dB

Dependencies:
    pip install tensorrt onnx onnxruntime-gpu numpy pillow torch torchvision
    # pycuda is NOT required — TRTRunner uses PyTorch CUDA tensors directly

TensorRT version note:
    This file targets TensorRT 8.x / 10.x via the unified `tensorrt` package.
    If you have TRT 8, the Builder API is used directly.
    If you have TRT 10+, the same API still works but you can also use
    `trtexec` on the CLI for a quicker engine build:

        trtexec --onnx=model.onnx --saveEngine=model_fp16.trt --fp16 \\
                --minShapes=input:1x3x32x32 \\
                --optShapes=input:8x3x64x64 \\
                --maxShapes=input:16x3x128x128

    Then point --trt_dir at the directory containing the .trt files.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully so the file is still importable
# on machines without TRT installed (e.g. AMD, Apple).
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    TRT_AVAILABLE = True
except (ImportError, AttributeError):
    TRT_AVAILABLE = False
    TRT_LOGGER = None
    logger.warning("TensorRT not found — export/TRT inference unavailable.")

# pycuda is not used — we use PyTorch CUDA tensors for H2D/D2H transfers.
# This avoids the cuCtxCreate_v4 / driver-version mismatch issues that
# pycuda commonly hits when the venv was built against a different CUDA toolkit.
PYCUDA_AVAILABLE = True   # kept for backward compat with any external callers

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not found — ORT baseline unavailable.")

try:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Re-use utilities from train.py rather than duplicating them
# ---------------------------------------------------------------------------

# Always use numpy implementations here — train.py's versions expect torch tensors.
def _make_blend_mask(size: int, overlap: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=np.float32)
    for i in range(overlap):
        w = 0.5 * (1.0 - math.cos(math.pi * i / overlap))
        mask[i,            :] = np.minimum(mask[i,            :], w)
        mask[size - 1 - i, :] = np.minimum(mask[size - 1 - i, :], w)
        mask[:,            i] = np.minimum(mask[:,            i], w)
        mask[:, size - 1 - i] = np.minimum(mask[:, size - 1 - i], w)
    return mask


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two numpy arrays in [0, 1]."""
    mse = float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


# ===========================================================================
# TensorRT engine builder
# ===========================================================================

def build_trt_engine(
    onnx_path:   str,
    engine_path: str,
    fp16:        bool = False,
    # Dynamic shape profile — tile sizes the engine will handle at runtime.
    # min / opt / max are (batch, C, H, W).
    # opt should match your most common tile batch size and spatial size.
    min_shape:   tuple[int, ...] = (1,  3, 32,  32),
    opt_shape:   tuple[int, ...] = (8,  3, 64,  64),
    max_shape:   tuple[int, ...] = (16, 3, 128, 128),
    workspace_gb: float = 2.0,
) -> bool:
    """
    Build a TensorRT engine from an ONNX file and serialise it to disk.

    Dynamic shapes are required because the C++ tiler sends variable tile
    counts per frame (edge tiles may be smaller) and we want to support
    different tile sizes without rebuilding the engine.

    Args:
        onnx_path:    Path to the source ONNX model.
        engine_path:  Where to write the .trt engine file.
        fp16:         Enable FP16 precision (requires Ampere or newer for
                      full benefit; Turing gets partial speedup).
        min/opt/max_shape: Dynamic shape profile (B, C, H, W).
        workspace_gb: GPU scratch memory for TRT kernel selection.

    Returns True on success.
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT not installed — cannot build engine.")
        return False

    precision = "FP16" if fp16 else "FP32"
    logger.info("Building TRT %s engine: %s → %s", precision, onnx_path, engine_path)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )

    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning(
                "GPU does not have fast FP16 support — engine will still "
                "build but may not be faster than FP32."
            )
        config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic shape profile — must cover min/opt/max for every input tensor.
    # Our ONNX has one input named 'input' with dynamic H and W axes.
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    logger.info(
        "Building engine (this takes 1-5 min on first run — TRT selects "
        "optimal CUDA kernels for your GPU and caches them in the .trt file)..."
    )
    serialised = builder.build_serialized_network(network, config)
    if serialised is None:
        logger.error("Engine build failed.")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialised)

    size_mb = os.path.getsize(engine_path) / 1e6
    logger.info("Engine saved → %s (%.1f MB)", engine_path, size_mb)
    return True


# ===========================================================================
# TensorRT inference runner
# ===========================================================================

class TRTRunner:
    """
    Runs inference on a serialised TRT engine using PyTorch CUDA tensors.

    No pycuda required — memory management is done via torch.cuda which
    uses the same CUDA context as PyTorch, avoiding the driver-version
    mismatch that pycuda commonly hits in virtualenvs.

    Usage:
        runner = TRTRunner("model_fp16.trt")
        output = runner.infer(batch_np)   # (N, 3, H, W) float32 in [0,1]
    """

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not installed.")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TRTRunner.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device found. TRTRunner requires a CUDA GPU."
            )

        with open(engine_path, "rb") as f:
            runtime       = trt.Runtime(TRT_LOGGER)
            self._engine  = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()

        self._device    = torch.device("cuda:0")
        # Cached tensors — reallocated if shape changes between calls
        self._in_tensor:  Optional[torch.Tensor] = None
        self._out_tensor: Optional[torch.Tensor] = None
        self._stream    = torch.cuda.Stream()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """
        Args:
            batch: float32 numpy array (N, 3, H, W), values in [0, 1].
        Returns:
            float32 numpy array (N, 3, H*scale, W*scale), values in [0, 1].
        """
        assert batch.dtype == np.float32, "Input must be float32"
        N, C, H, W = batch.shape

        self._context.set_input_shape("input", (N, C, H, W))

        out_shape = tuple(self._context.get_tensor_shape("output"))
        out_size  = int(np.prod(out_shape))

        with torch.cuda.stream(self._stream):
            # H2D: numpy → pinned host tensor → CUDA device tensor
            in_tensor = torch.from_numpy(
                np.ascontiguousarray(batch)
            ).pin_memory().to(self._device, non_blocking=True)

            # Allocate output tensor on device (reuse if shape matches)
            if (self._out_tensor is None
                    or self._out_tensor.numel() != out_size):
                self._out_tensor = torch.empty(
                    out_size, dtype=torch.float32, device=self._device
                )

            # Give TRT the raw device pointers — no pycuda involved
            self._context.set_tensor_address(
                "input",  in_tensor.data_ptr()
            )
            self._context.set_tensor_address(
                "output", self._out_tensor.data_ptr()
            )
            self._context.execute_async_v3(
                stream_handle=self._stream.cuda_stream
            )

        # Sync and D2H
        self._stream.synchronize()
        return self._out_tensor.reshape(out_shape).cpu().numpy()


# ===========================================================================
# ORT inference runner (FP32 baseline)
# ===========================================================================

class ORTRunner:
    """ORT inference runner — FP32 baseline via CUDAExecutionProvider."""

    def __init__(self, onnx_path: str):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self._session    = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ORT providers active: %s",
                    [p.name for p in self._session.get_providers()])

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return self._session.run(None, {self._input_name: batch})[0]


# ===========================================================================
# Tiled inference (shared between ORT and TRT runners)
# ===========================================================================

def infer_tiled(
    runner,           # ORTRunner | TRTRunner — anything with .infer(np.ndarray)
    image_np: np.ndarray,     # (3, H, W) float32 [0,1] — full LR frame
    scale:    int,
    hr_tile:  int,            # output tile size (HR pixels)
    overlap:  int  = 16,
    batch_sz: int  = 8,       # tiles per forward pass
) -> np.ndarray:
    """
    Tile a full LR frame, run SR on each tile (in batches), and stitch.

    Returns (3, H*scale, W*scale) float32 array.
    """
    C, lH, lW = image_np.shape
    lr_tile = hr_tile // scale
    step    = lr_tile - 2 * (overlap // scale)
    assert step > 0, "overlap too large relative to tile size"

    oH, oW  = lH * scale, lW * scale
    accum   = np.zeros((C, oH, oW), dtype=np.float32)
    weight  = np.zeros((1, oH, oW), dtype=np.float32)
    mask_hr = _make_blend_mask(hr_tile, overlap)

    # Collect all tile positions
    positions = []
    y = 0
    while y < lH:
        x = 0
        while x < lW:
            ty = min(y, lH - lr_tile)
            tx = min(x, lW - lr_tile)
            positions.append((ty, tx))
            x += step
        y += step

    # Process in batches
    for i in range(0, len(positions), batch_sz):
        batch_pos  = positions[i:i + batch_sz]
        tiles      = []
        for ty, tx in batch_pos:
            tile = image_np[:, ty:ty + lr_tile, tx:tx + lr_tile]
            if tile.shape[1] < lr_tile or tile.shape[2] < lr_tile:
                # Pad edge tiles
                pad = np.zeros((C, lr_tile, lr_tile), dtype=np.float32)
                pad[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = pad
            tiles.append(tile)

        batch_np  = np.stack(tiles, axis=0)  # (B, C, lr_tile, lr_tile)
        out_batch = runner.infer(batch_np)    # (B, C, hr_tile, hr_tile)
        out_batch = np.clip(out_batch, 0.0, 1.0)

        for k, (ty, tx) in enumerate(batch_pos):
            oy, ox = ty * scale, tx * scale
            ch = min(hr_tile, oH - oy)
            cw = min(hr_tile, oW - ox)
            accum [:, oy:oy+ch, ox:ox+cw] += out_batch[k, :, :ch, :cw] * mask_hr[:ch, :cw]
            weight[:,  oy:oy+ch, ox:ox+cw] += mask_hr[:ch, :cw]

    return np.clip(accum / np.maximum(weight, 1e-6), 0.0, 1.0)


# ===========================================================================
# Benchmark
# ===========================================================================

def _load_test_images(
    data_dir: str,
    sr_rate:  int,
    hr_size:  int,
    n:        int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load n images, return list of (lr_np, hr_np) pairs.
    lr_np: (3, H//sr_rate, W//sr_rate) float32 [0,1]
    hr_np: (3, H, W)                   float32 [0,1]
    """
    files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ][:n]

    pairs = []
    for fname in files:
        img = Image.open(os.path.join(data_dir, fname)).convert("RGB")
        w, h = img.size
        # Crop to whole tile multiples
        nw, nh = w // hr_size, h // hr_size
        if nw == 0 or nh == 0:
            continue
        img    = img.crop((0, 0, nw * hr_size, nh * hr_size))
        lr_img = img.resize((img.width // sr_rate, img.height // sr_rate),
                            Image.LANCZOS)

        hr_np = np.array(img,    dtype=np.float32).transpose(2,0,1) / 255.0
        lr_np = np.array(lr_img, dtype=np.float32).transpose(2,0,1) / 255.0
        pairs.append((lr_np, hr_np))

    return pairs


def _benchmark_runner(
    runner,
    pairs:    list,
    sr_rate:  int,
    hr_tile:  int,
    overlap:  int,
    batch_sz: int,
    warmup:   int = 3,
) -> tuple[float, float]:
    """Returns (avg_fps, avg_psnr) over all pairs."""
    total_psnr = 0.0
    total_time = 0.0
    n_images   = 0

    for i, (lr_np, hr_np) in enumerate(pairs):
        # Warmup on first image
        if i == 0:
            for _ in range(warmup):
                _ = infer_tiled(runner, lr_np, sr_rate, hr_tile, overlap, batch_sz)

        t0     = time.perf_counter()
        sr_np  = infer_tiled(runner, lr_np, sr_rate, hr_tile, overlap, batch_sz)
        elapsed = time.perf_counter() - t0

        total_time += elapsed
        total_psnr += calculate_psnr(hr_np, sr_np)
        n_images   += 1

    avg_fps  = n_images / total_time
    avg_psnr = total_psnr / n_images
    return avg_fps, avg_psnr


def run_benchmark(
    onnx_path:  str,
    trt_fp32:   Optional[str],
    trt_fp16:   Optional[str],
    data_dir:   str,
    sr_rate:    int  = 4,
    hr_size:    int  = 192,   # full-image HR tile size (for data loading)
    hr_tile:    int  = 256,   # SR output tile size for inference
    overlap:    int  = 16,
    batch_sz:   int  = 8,
    n_images:   int  = 5,
) -> None:
    """
    Run all available backends and print a comparison table.

    hr_tile controls the SR output patch size during inference — this
    should match what the C++ tiler will use. A 256×256 HR output tile
    means 64×64 LR input at 4x scale.
    """
    logger.info("Loading test images from %s", data_dir)
    pairs = _load_test_images(data_dir, sr_rate, hr_size, n=n_images)
    if not pairs:
        logger.error("No valid images found in %s", data_dir)
        return
    logger.info("Loaded %d images for benchmark", len(pairs))

    results: list[tuple[str, float, float, str]] = []  # (name, fps, psnr, note)

    # --- ORT FP32 ---
    if ORT_AVAILABLE and onnx_path:
        logger.info("Running ORT FP32...")
        try:
            runner = ORTRunner(onnx_path)
            fps, psnr = _benchmark_runner(
                runner, pairs, sr_rate, hr_tile, overlap, batch_sz
            )
            results.append(("ORT FP32", fps, psnr, "baseline"))
        except Exception as e:
            logger.warning("ORT FP32 failed: %s", e)

    # --- TRT FP32 ---
    if TRT_AVAILABLE and PYCUDA_AVAILABLE and trt_fp32 and os.path.exists(trt_fp32):
        logger.info("Running TRT FP32...")
        try:
            runner = TRTRunner(trt_fp32)
            fps, psnr = _benchmark_runner(
                runner, pairs, sr_rate, hr_tile, overlap, batch_sz
            )
            ort_fps = results[0][1] if results else None
            note = (f"~{fps/ort_fps:.1f}x vs ORT" if ort_fps else "")
            results.append(("TRT FP32", fps, psnr, note))
        except Exception as e:
            logger.warning("TRT FP32 failed: %s", e)

    # --- TRT FP16 ---
    if TRT_AVAILABLE and PYCUDA_AVAILABLE and trt_fp16 and os.path.exists(trt_fp16):
        logger.info("Running TRT FP16...")
        try:
            runner = TRTRunner(trt_fp16)
            fps, psnr = _benchmark_runner(
                runner, pairs, sr_rate, hr_tile, overlap, batch_sz
            )
            ort_fps  = results[0][1] if results else None
            ort_psnr = results[0][2] if results else None
            speedup  = f"~{fps/ort_fps:.1f}x vs ORT" if ort_fps else ""
            psnr_delta = (
                f", {psnr - ort_psnr:+.2f} dB vs ORT" if ort_psnr else ""
            )
            results.append(("TRT FP16", fps, psnr, speedup + psnr_delta))
        except Exception as e:
            logger.warning("TRT FP16 failed: %s", e)

    if not results:
        logger.error("No backends ran successfully.")
        return

    # --- Print table ---
    col_w = [18, 16, 12, 30]
    header = (
        f"  {'Backend':<{col_w[0]}} "
        f"{'FPS (batch-8)':>{col_w[1]}} "
        f"{'PSNR (dB)':>{col_w[2]}} "
        f"{'Notes':<{col_w[3]}}"
    )
    sep = "  " + "─" * (sum(col_w) + 3)

    print(f"\n{sep}")
    print(f"  Benchmark  |  sr_rate={sr_rate}  hr_tile={hr_tile}×{hr_tile}"
          f"  batch={batch_sz}  images={len(pairs)}")
    print(sep)
    print(header)
    print(sep)
    for name, fps, psnr, note in results:
        print(
            f"  {name:<{col_w[0]}} "
            f"{fps:>{col_w[1]}.1f} "
            f"{psnr:>{col_w[2]}.2f} "
            f"{note:<{col_w[3]}}"
        )
    print(sep)
    print(
        "\n  PSNR is measured against the original HR image.\n"
        "  A delta < 0.1 dB between backends is perceptually lossless.\n"
        "  FPS counts full images per second (all tiles, blended).\n"
    )


# ===========================================================================
# CLI
# ===========================================================================

def _cmd_export(args) -> None:
    if not TRT_AVAILABLE:
        logger.error(
            "TensorRT is not installed.\n"
            "Install it from https://developer.nvidia.com/tensorrt\n"
            "or use trtexec directly:\n"
            "  trtexec --onnx=model.onnx --saveEngine=model_fp16.trt --fp16 \\\n"
            "          --minShapes=input:1x3x32x32 \\\n"
            "          --optShapes=input:8x3x64x64 \\\n"
            "          --maxShapes=input:16x3x128x128"
        )
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    stem     = Path(args.onnx).stem
    fp32_out = os.path.join(args.out_dir, f"{stem}_fp32.trt")
    fp16_out = os.path.join(args.out_dir, f"{stem}_fp16.trt")

    # Parse dynamic shape args
    def parse_shape(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split(","))

    min_s = parse_shape(args.min_shape)
    opt_s = parse_shape(args.opt_shape)
    max_s = parse_shape(args.max_shape)

    logger.info("Building FP32 engine...")
    build_trt_engine(
        args.onnx, fp32_out,
        fp16=False,
        min_shape=min_s, opt_shape=opt_s, max_shape=max_s,
        workspace_gb=args.workspace,
    )

    logger.info("Building FP16 engine...")
    build_trt_engine(
        args.onnx, fp16_out,
        fp16=True,
        min_shape=min_s, opt_shape=opt_s, max_shape=max_s,
        workspace_gb=args.workspace,
    )

    print(f"\nEngines written to {args.out_dir}/")
    print(f"  {os.path.basename(fp32_out)}")
    print(f"  {os.path.basename(fp16_out)}")
    print(
        "\nTo benchmark them:\n"
        f"  python quant.py bench --onnx {args.onnx} "
        f"--trt_dir {args.out_dir} "
        f"--data_dir <your_images> --sr_rate {args.sr_rate}"
    )


def _cmd_bench(args) -> None:
    stem     = Path(args.onnx).stem
    trt_fp32 = os.path.join(args.trt_dir, f"{stem}_fp32.trt") if args.trt_dir else None
    trt_fp16 = os.path.join(args.trt_dir, f"{stem}_fp16.trt") if args.trt_dir else None

    run_benchmark(
        onnx_path=args.onnx,
        trt_fp32=trt_fp32,
        trt_fp16=trt_fp16,
        data_dir=args.data_dir,
        sr_rate=args.sr_rate,
        hr_size=args.hr_size,
        hr_tile=args.hr_tile,
        overlap=args.overlap,
        batch_sz=args.batch_sz,
        n_images=args.n_images,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorRT export and inference benchmark for anime SR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- export subcommand ---
    exp = sub.add_parser("export", help="Build TRT FP32 and FP16 engines from ONNX")
    exp.add_argument("--onnx",      required=True, help="Input ONNX model path")
    exp.add_argument("--out_dir",   default="trt_engines",
                     help="Output directory for .trt files (default: trt_engines/)")
    exp.add_argument("--sr_rate",   type=int, default=4)
    exp.add_argument("--workspace", type=float, default=2.0,
                     help="TRT workspace in GB (default: 2.0)")
    # Dynamic shape profile: comma-separated B,C,H,W
    exp.add_argument("--min_shape", default="1,3,32,32",
                     help="Min input shape B,C,H,W (default: 1,3,32,32)")
    exp.add_argument("--opt_shape", default="8,3,64,64",
                     help="Opt input shape B,C,H,W (default: 8,3,64,64)")
    exp.add_argument("--max_shape", default="16,3,128,128",
                     help="Max input shape B,C,H,W (default: 16,3,128,128)")

    # --- bench subcommand ---
    ben = sub.add_parser("bench", help="Benchmark ORT vs TRT FP32 vs TRT FP16")
    ben.add_argument("--onnx",     required=True, help="ONNX model path (for ORT baseline)")
    ben.add_argument("--trt_dir",  default=None,
                     help="Directory containing .trt files built by 'export'")
    ben.add_argument("--data_dir", required=True, help="Directory of HR test images")
    ben.add_argument("--sr_rate",  type=int, default=4)
    ben.add_argument("--hr_size",  type=int, default=192,
                     help="HR patch size used to load images (default: 192)")
    ben.add_argument("--hr_tile",  type=int, default=256,
                     help="SR output tile size for inference (default: 256). "
                          "LR input tile = hr_tile // sr_rate.")
    ben.add_argument("--overlap",  type=int, default=16,
                     help="Overlap between tiles in HR pixels (default: 16)")
    ben.add_argument("--batch_sz", type=int, default=8,
                     help="Tiles per forward pass (default: 8)")
    ben.add_argument("--n_images", type=int, default=5,
                     help="Number of test images to use (default: 5)")

    args = parser.parse_args()
    if args.cmd == "export":
        _cmd_export(args)
    elif args.cmd == "bench":
        _cmd_bench(args)


if __name__ == "__main__":
    main()
