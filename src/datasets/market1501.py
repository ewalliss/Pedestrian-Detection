"""Market-1501 dataset loader.

Filename format: {pid:04d}_c{cam}s{seq}_{frame:06d}_{det:02d}.jpg
  pid  : person identity (1-1501; 0 = junk)
  cam  : camera id (1-6)
  seq  : sequence number (ignored in this pipeline)
  frame: frame index (ignored)
  det  : detection index (ignored)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

_FNAME_RE = re.compile(r"^(\d{4})_c(\d)s\d_\d{6}_\d{2}\.jpg$")


@dataclass
class ReIDSample:
    img_path: Path
    pid: int       # global, 0-indexed after remapping
    cam_id: int    # 1-indexed camera
    view_id: int = 0
    source: str = "market1501"


def _parse_fname(fname: str) -> tuple[int, int] | None:
    """Return (pid, cam_id) or None if filename does not match."""
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    pid, cam = int(m.group(1)), int(m.group(2))
    if pid == 0:   # junk / distractor
        return None
    return pid, cam


class Market1501(Dataset):
    """Market-1501 dataset for a single split.

    Args:
        root:      Path to Market-1501-v15.09.15 directory.
        split:     "train", "query", or "gallery".
        transform: Optional image transform.
        mini:      If True, limit to the first `mini_num_ids` identities.
        mini_num_ids: Number of identities in mini mode.
    """

    _SPLIT_DIRS = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
    }

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transform: Callable | None = None,
        mini: bool = False,
        mini_num_ids: int = 100,
        remap_pids: bool = True,
    ) -> None:
        root = Path(root)
        if split not in self._SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(self._SPLIT_DIRS)}, got '{split}'")
        img_dir = root / self._SPLIT_DIRS[split]
        if not img_dir.exists():
            raise FileNotFoundError(f"Market-1501 directory not found: {img_dir}")

        self.transform = transform
        self._samples: list[ReIDSample] = []

        raw: list[tuple[int, int, Path]] = []  # (raw_pid, cam_id, path)
        for p in sorted(img_dir.glob("*.jpg")):
            parsed = _parse_fname(p.name)
            if parsed is None:
                continue
            raw_pid, cam = parsed
            raw.append((raw_pid, cam, p))

        if not raw:
            raise RuntimeError(f"No valid images found in {img_dir}")

        # Mini mode: keep only the first mini_num_ids unique identities
        if mini:
            unique_pids = sorted({r[0] for r in raw})[:mini_num_ids]
            pid_set = set(unique_pids)
            raw = [r for r in raw if r[0] in pid_set]

        unique_sorted = sorted({r[0] for r in raw})
        if remap_pids:
            # Contiguous 0-indexed mapping — required for train classifier
            pid_map = {p: i for i, p in enumerate(unique_sorted)}
        else:
            # Identity mapping — preserves raw file PIDs so query/gallery match
            pid_map = {p: p for p in unique_sorted}

        for raw_pid, cam, path in raw:
            self._samples.append(
                ReIDSample(
                    img_path=path,
                    pid=pid_map[raw_pid],
                    cam_id=cam,
                )
            )

        self.num_pids = len(unique_sorted)
        # Use max_cam_id + 1 so nn.Embedding table covers all 1-indexed cam IDs
        self.num_cams = max(s.cam_id for s in self._samples) + 1

    # ── Required by RandomIdentitySampler ─────────────────────────────────────
    @property
    def pids(self) -> list[int]:
        return [s.pid for s in self._samples]

    # ── Dataset interface ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        s = self._samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.pid, s.cam_id, s.view_id
