import os
import random
import logging

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


def _scan_images(root_dir: str, min_size: int) -> list[str]:
    """
    Walk root_dir, return paths of images that:
      1. Have a recognised extension
      2. Can actually be opened by Pillow (skips corrupt files)
      3. Are at least min_size × min_size pixels

    Logs a warning for every file that fails either check so you know what
    is being silently dropped from training.
    """
    candidates = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    valid = []
    for path in candidates:
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    logger.warning(
                        "Skipping %s — too small (%dx%d, need %d)",
                        path, w, h, min_size,
                    )
                    continue
        except (UnidentifiedImageError, OSError) as e:
            logger.warning("Skipping %s — cannot open (%s)", path, e)
            continue
        valid.append(path)

    return valid


class SRDataset(Dataset):
    """
    Patch-based super-resolution dataset.

    For each sample:
      1. Randomly crop an hr_size × hr_size patch from the HR image.
      2. Apply spatial augmentations (hflip, vflip, 90° rotations).
      3. Optionally apply mild colour jitter (hue/saturation only — no
         brightness changes that would corrupt anime flat shading).
      4. Downsample the HR patch to LR using the chosen filter.

    Args:
        root_dir:      Directory of HR images.
        hr_size:       Spatial size of HR patches (must be divisible by scale).
        sr_rate:       Upscale factor (LR = HR / sr_rate).
        downsample:    'bicubic' or 'lanczos'. LANCZOS preserves anime line
                       sharpness better in the LR input; BICUBIC is faster.
        color_jitter:  If True, apply mild hue/saturation jitter.
    """

    RESAMPLE = {
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }

    def __init__(
        self,
        root_dir:     str,
        hr_size:      int  = 192,
        sr_rate:      int  = 4,
        downsample:   str  = 'lanczos',
        color_jitter: bool = True,
    ):
        assert hr_size % sr_rate == 0, (
            f"hr_size ({hr_size}) must be divisible by sr_rate ({sr_rate})"
        )
        assert downsample in self.RESAMPLE, (
            f"downsample must be one of {list(self.RESAMPLE)}"
        )

        self.hr_size    = hr_size
        self.scale      = sr_rate
        self.resample   = self.RESAMPLE[downsample]

        # Colour jitter: hue ±5°, saturation ±20% — conservative for anime
        self.jitter = (
            T.ColorJitter(hue=0.05, saturation=0.2)
            if color_jitter else None
        )

        self.paths = _scan_images(root_dir, min_size=hr_size)
        if not self.paths:
            raise RuntimeError(
                f"No valid images >= {hr_size}px found in {root_dir!r}"
            )
        logger.info("Dataset: %d images in %r", len(self.paths), root_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        try:
            hr_img = Image.open(path).convert('RGB')
        except (OSError, UnidentifiedImageError):
            # Corrupt file appeared after init scan; return a random other sample
            logger.warning("Failed to open %s at getitem, substituting.", path)
            return self[random.randint(0, len(self) - 1)]

        w, h = hr_img.size

        # --- Random crop ---
        left = random.randint(0, w - self.hr_size)
        top  = random.randint(0, h - self.hr_size)
        hr_img = hr_img.crop((left, top, left + self.hr_size, top + self.hr_size))

        # --- Spatial augmentations ---
        if random.random() > 0.5:
            hr_img = F.hflip(hr_img)
        if random.random() > 0.5:
            hr_img = F.vflip(hr_img)
        # Random 90° rotations (angle ∈ {0, 90, 180, 270})
        k = random.randint(0, 3)
        if k:
            hr_img = F.rotate(hr_img, 90 * k)

        # --- Colour jitter (HR only — LR inherits via downsampling) ---
        if self.jitter is not None:
            hr_img = self.jitter(hr_img)

        # --- Downsample to LR ---
        lr_size = self.hr_size // self.scale
        lr_img  = hr_img.resize((lr_size, lr_size), self.resample)

        return T.ToTensor()(lr_img), T.ToTensor()(hr_img)
