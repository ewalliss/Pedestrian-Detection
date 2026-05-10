"""PedestrianDataset — merges Market-1501 train with optional MOT crops.

For the mini pipeline, only Market-1501 is used (MOT extraction is skipped).
For full training, MOT crops are merged after extraction via mot_extractor.py.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from .market1501 import Market1501, ReIDSample
from .samplers import RandomIdentitySampler
from .transforms import build_train_transform, build_val_transform


class PedestrianDataset(Dataset):
    """Unified training dataset: Market-1501 + optional MOT crops.

    Args:
        samples:   List of ReIDSample (all splits merged and re-indexed).
        transform: Image transform applied per __getitem__.
    """

    def __init__(self, samples: list[ReIDSample], transform: Callable | None = None) -> None:
        self._samples = samples
        self.transform = transform
        self.num_pids = len({s.pid for s in samples})
        # Use max_cam_id + 1 so nn.Embedding table covers all 1-indexed cam IDs
        self.num_cams = max(s.cam_id for s in samples) + 1

    @property
    def pids(self) -> list[int]:
        return [s.pid for s in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        from PIL import Image
        s = self._samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.pid, s.cam_id, s.view_id


def build_pedestrian_loaders(
    market1501_root: Path,
    batch_size: int,
    num_instances: int,
    num_workers: int,
    mini: bool = False,
    mini_num_ids: int = 100,
    mot_crops_root: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Build train / query / gallery DataLoaders.

    Returns:
        (train_loader, query_loader, gallery_loader, num_train_pids, num_train_cams)
    """
    train_tf = build_train_transform()
    val_tf = build_val_transform()

    market_train = Market1501(market1501_root, split="train", mini=mini, mini_num_ids=mini_num_ids)
    # remap_pids=False preserves raw file pids so query/gallery pids are
    # comparable across splits (evaluate.py matches by raw pid equality).
    market_query = Market1501(market1501_root, split="query", transform=val_tf, remap_pids=False)
    market_gallery = Market1501(market1501_root, split="gallery", transform=val_tf, remap_pids=False)

    # Merge train samples (Market-1501 + MOT crops if provided)
    train_samples = copy.deepcopy(market_train._samples)

    if mot_crops_root is not None and mot_crops_root.exists():
        train_samples.extend(_load_mot_crops(mot_crops_root, base_pid_offset=market_train.num_pids))

    # Re-index pids contiguously across merged dataset
    all_pids = sorted({s.pid for s in train_samples})
    pid_remap = {p: i for i, p in enumerate(all_pids)}
    for s in train_samples:
        s.pid = pid_remap[s.pid]

    train_dataset = PedestrianDataset(train_samples, transform=train_tf)

    sampler = RandomIdentitySampler(train_dataset, batch_size=batch_size, num_instances=num_instances)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    query_loader = DataLoader(
        market_query,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        market_gallery,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        query_loader,
        gallery_loader,
        train_dataset.num_pids,
        train_dataset.num_cams,
    )


def _load_mot_crops(root: Path, base_pid_offset: int) -> list[ReIDSample]:
    """Walk data/mot_crops/{cam_id}/{pid}/ and return ReIDSample list."""
    samples = []
    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam_id = int(cam_dir.name)
        for pid_dir in sorted(cam_dir.iterdir()):
            if not pid_dir.is_dir():
                continue
            pid = int(pid_dir.name) + base_pid_offset
            for img_path in sorted(pid_dir.glob("*.jpg")):
                samples.append(ReIDSample(img_path=img_path, pid=pid, cam_id=cam_id, source="mot"))
    return samples
