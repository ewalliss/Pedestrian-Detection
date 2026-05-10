"""Image transforms for ReID training and validation."""

from __future__ import annotations

import math
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

_CLIP_MEAN = (0.5, 0.5, 0.5)
_CLIP_STD = (0.5, 0.5, 0.5)
_INPUT_SIZE = (224, 224)  # CLIP ViT-B/16 native resolution


class RandomErasing:
    """Randomly erase a rectangular patch of an image tensor.

    Args:
        prob: Probability of applying erasing.
        sl, sh: Area fraction range of erased region.
        r1, r2: Aspect ratio range of erased region.
        mean: Fill value per channel.
    """

    def __init__(
        self,
        prob: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        r2: float = 3.33,
        mean: tuple[float, float, float] = _CLIP_MEAN,
    ) -> None:
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.mean = mean

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return img
        _, h, w = img.shape
        area = h * w
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect = math.exp(random.uniform(math.log(self.r1), math.log(self.r2)))
            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))
            if ew < w and eh < h:
                x1 = random.randint(0, w - ew)
                y1 = random.randint(0, h - eh)
                for c, m in enumerate(self.mean):
                    img[c, y1 : y1 + eh, x1 : x1 + ew] = m
                return img
        return img


def build_train_transform(input_size: tuple[int, int] = _INPUT_SIZE) -> T.Compose:
    """Augmentation pipeline for training."""
    h, w = input_size
    pad = 10
    return T.Compose(
        [
            T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(pad),
            T.RandomCrop((h, w)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            RandomErasing(prob=0.5),
        ]
    )


def build_val_transform(input_size: tuple[int, int] = _INPUT_SIZE) -> T.Compose:
    """Minimal transform for evaluation — no augmentation."""
    h, w = input_size
    return T.Compose(
        [
            T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ]
    )
