#!/usr/bin/env python3
"""
compare.py — Visual + quantitative comparison of bicubic vs APISR upscaling.

Takes a high-resolution anime image, downscales it to create an LR input,
then upscales that LR input two ways:
  1. Plain bicubic interpolation (PIL/OpenCV baseline)
  2. Your trained APISR model (via ONNX)

Saves a side-by-side comparison image and prints PSNR for both, so you can
see exactly what the model is adding over naive upscaling.

Usage:
    python compare.py \
        --image /path/to/anime_image.png \
        --onnx model/apisr_2x.onnx \
        --sr_rate 2 \
        --output comparison.png

    # Crop a specific region for close-up comparison (line art, text, etc.)
    python compare.py \
        --image /path/to/anime_image.png \
        --onnx model/apisr_2x.onnx \
        --sr_rate 2 \
        --crop 256 \
        --output comparison_crop.png
"""
import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed — run: pip install onnxruntime-gpu")
    exit(1)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def make_blend_mask(size: int, overlap: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=np.float32)
    for i in range(overlap):
        w = 0.5 * (1.0 - math.cos(math.pi * i / overlap))
        mask[i, :] = np.minimum(mask[i, :], w)
        mask[size - 1 - i, :] = np.minimum(mask[size - 1 - i, :], w)
        mask[:, i] = np.minimum(mask[:, i], w)
        mask[:, size - 1 - i] = np.minimum(mask[:, size - 1 - i], w)
    return mask


def sr_tiled(session, lr_np, scale, overlap=8):
    """Run tiled SR inference on a full LR image via ONNX."""
    input_name = session.get_inputs()[0].name
    C, H, W = lr_np.shape

    # Read the expected tile size from the model's input shape.
    # If the model has dynamic axes, the shape will be something like
    # ['batch', 3, 'height', 'width'] with string placeholders -- fall back
    # to 64 in that case. If fixed, use the actual value.
    input_shape = session.get_inputs()[0].shape
    if isinstance(input_shape[2], int) and input_shape[2] > 0:
        lr_tile = input_shape[2]
    else:
        lr_tile = 64  # dynamic axes — default to 64

    hr_tile = lr_tile * scale
    step = lr_tile - 2 * (overlap // scale)
    if step <= 0:
        step = lr_tile // 2  # fallback if overlap is too large

    print(f"  Tiling: LR tile={lr_tile}x{lr_tile}, HR tile={hr_tile}x{hr_tile}, "
          f"step={step}, overlap={overlap}")


    oH, oW = H * scale, W * scale
    accum = np.zeros((C, oH, oW), dtype=np.float32)
    weight = np.zeros((1, oH, oW), dtype=np.float32)
    mask = make_blend_mask(hr_tile, overlap)

    positions = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            ty = min(y, max(0, H - lr_tile))
            tx = min(x, max(0, W - lr_tile))
            positions.append((ty, tx))
            x += step
        y += step

    # Process in batches
    batch_size = 16
    for i in range(0, len(positions), batch_size):
        batch_pos = positions[i:i + batch_size]
        tiles = []
        for ty, tx in batch_pos:
            tile = lr_np[:, ty:ty + lr_tile, tx:tx + lr_tile]
            pad = np.zeros((C, lr_tile, lr_tile), dtype=np.float32)
            pad[:, :tile.shape[1], :tile.shape[2]] = tile
            tiles.append(pad)

        batch = np.stack(tiles, axis=0)
        out_batch = session.run(None, {input_name: batch})[0]
        out_batch = np.clip(out_batch, 0.0, 1.0)

        for k, (ty, tx) in enumerate(batch_pos):
            oy, ox = ty * scale, tx * scale
            ch = min(hr_tile, oH - oy)
            cw = min(hr_tile, oW - ox)
            accum[:, oy:oy+ch, ox:ox+cw] += out_batch[k, :, :ch, :cw] * mask[:ch, :cw]
            weight[:, oy:oy+ch, ox:ox+cw] += mask[:ch, :cw]

    return np.clip(accum / np.maximum(weight, 1e-6), 0.0, 1.0)


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Add a label to the bottom of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = img.height - th - 10

    # Draw background rectangle for readability
    draw.rectangle([x - 5, y - 3, x + tw + 5, y + th + 3], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Compare bicubic vs APISR upscaling side-by-side"
    )
    parser.add_argument("--image", required=True, help="HR anime image to test")
    parser.add_argument("--onnx", required=True, help="Trained ONNX model")
    parser.add_argument("--sr_rate", type=int, default=2, help="Scale factor (2 or 4)")
    parser.add_argument("--crop", type=int, default=0,
                        help="If set, crop a center patch of this size (HR pixels) "
                             "for a zoomed comparison. 0 = full image.")
    parser.add_argument("--output", default="comparison.png", help="Output path")
    parser.add_argument("--overlap", type=int, default=8, help="Tile overlap in HR pixels")
    args = parser.parse_args()

    scale = args.sr_rate

    # Load the ONNX model first — we need it to detect the expected tile size
    session = ort.InferenceSession(
        args.onnx,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Load image and prepare LR/HR pair
    hr_img = Image.open(args.image).convert("RGB")
    w, h = hr_img.size

    # Read the model's expected LR tile size for clean cropping
    input_shape = session.get_inputs()[0].shape
    if isinstance(input_shape[2], int) and input_shape[2] > 0:
        lr_tile = input_shape[2]
    else:
        lr_tile = 64
    block = lr_tile * scale
    w_crop = (w // block) * block
    h_crop = (h // block) * block
    if w_crop == 0 or h_crop == 0:
        print(f"Image too small ({w}x{h}) — need at least {block}x{block}")
        return
    hr_img = hr_img.crop((0, 0, w_crop, h_crop))

    # Create LR by downscaling (simulates the real-world input)
    lr_w, lr_h = w_crop // scale, h_crop // scale
    lr_img = hr_img.resize((lr_w, lr_h), Image.LANCZOS)

    # Method 1: Bicubic upscale (the baseline)
    bicubic_img = lr_img.resize((w_crop, h_crop), Image.BICUBIC)

    # Method 2: APISR model via ONNX
    print(f"Running APISR ({args.onnx}) on {lr_w}x{lr_h} input...")
    lr_np = np.array(lr_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    sr_np = sr_tiled(session, lr_np, scale, overlap=args.overlap)
    sr_img = Image.fromarray(
        (sr_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    )

    # Compute PSNR against the original HR
    hr_np = np.array(hr_img, dtype=np.float32) / 255.0
    bicubic_np = np.array(bicubic_img, dtype=np.float32) / 255.0
    sr_display_np = np.array(sr_img, dtype=np.float32) / 255.0

    psnr_bicubic = calculate_psnr(hr_np, bicubic_np)
    psnr_sr = calculate_psnr(hr_np, sr_display_np)

    print(f"\nResults ({lr_w}x{lr_h} -> {w_crop}x{h_crop}, {scale}x):")
    print(f"  Bicubic PSNR:  {psnr_bicubic:.2f} dB")
    print(f"  APISR PSNR:    {psnr_sr:.2f} dB")
    print(f"  Improvement:   {psnr_sr - psnr_bicubic:+.2f} dB")

    # Optional center crop for zoomed comparison
    if args.crop > 0:
        cs = args.crop
        cx, cy = w_crop // 2 - cs // 2, h_crop // 2 - cs // 2
        hr_img = hr_img.crop((cx, cy, cx + cs, cy + cs))
        bicubic_img = bicubic_img.crop((cx, cy, cx + cs, cy + cs))
        sr_img = sr_img.crop((cx, cy, cx + cs, cy + cs))

    # Build side-by-side: Original HR | Bicubic | APISR
    panel_w, panel_h = hr_img.size
    gap = 4
    canvas = Image.new("RGB", (panel_w * 3 + gap * 2, panel_h), (40, 40, 40))
    canvas.paste(hr_img, (0, 0))
    canvas.paste(bicubic_img, (panel_w + gap, 0))
    canvas.paste(sr_img, (panel_w * 2 + gap * 2, 0))

    # Labels
    add_label(canvas.crop((0, 0, panel_w, panel_h)), "Original HR")
    # Re-paste labeled versions
    labeled_hr = hr_img.copy()
    labeled_bicubic = bicubic_img.copy()
    labeled_sr = sr_img.copy()
    add_label(labeled_hr, "Original HR")
    add_label(labeled_bicubic, f"Bicubic ({psnr_bicubic:.1f} dB)")
    add_label(labeled_sr, f"APISR ({psnr_sr:.1f} dB)")

    canvas.paste(labeled_hr, (0, 0))
    canvas.paste(labeled_bicubic, (panel_w + gap, 0))
    canvas.paste(labeled_sr, (panel_w * 2 + gap * 2, 0))

    canvas.save(args.output)
    print(f"\nSaved comparison -> {args.output}")
    print(f"  Left: Original HR | Middle: Bicubic {scale}x | Right: APISR {scale}x")


if __name__ == "__main__":
    main()
