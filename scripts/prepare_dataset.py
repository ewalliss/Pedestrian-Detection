"""Unified pedestrian ReID dataset preparation.

Converts Market-1501, MCC_pedestrian, and Wildtrack into a single
Market-1501-format dataset under a target ``Dataset/`` directory.

Target filename format (Market-1501):
    {pid:04d}_c{cam}s1_{frame:06d}_{det:02d}.jpg

PID namespace (no collisions, all ≤ 4 digits):
    Market-1501   :     1 –  1501   (kept as-is)
    MCC_pedestrian: 2001 –  9140   (+2000 offset)
    Wildtrack     : 9201 –  9513   (sequential remap of 313 IDs + 9200)

Camera IDs (1-indexed, single digit):
    Market-1501   : 1-6   (kept as-is)
    MCC_pedestrian: 1-8   (kept as-is)
    Wildtrack     : 1-7   (viewNum 0-6 → cam 1-7)

Output splits:
    bounding_box_train  — Market train + MCC train + ALL Wildtrack crops
    query               — Market query + MCC query
    bounding_box_test   — Market gallery + MCC gallery

Usage:
    python scripts/prepare_dataset.py \\
        --src  /path/to/pedestrian \\
        --dst  /path/to/project/Dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ── PID offsets ───────────────────────────────────────────────────────────────
_MCC_PID_OFFSET = 2000
_WILDTRACK_PID_OFFSET = 9200

# ── Filename regexes ──────────────────────────────────────────────────────────
_MARKET_RE = re.compile(r"^(\d{4})_c(\d)s\d_\d{6}_\d{2}\.jpg$")
_MCC_RE = re.compile(r"^(\d{4})_c(\d+)_f(\d+)\.jpg$")

# Minimum crop dimension to skip degenerate bboxes
_MIN_CROP_PX = 16


# ── Market-1501 ───────────────────────────────────────────────────────────────

def _copy_market(src: Path, dst: Path) -> None:
    """Copy Market-1501 splits to dst, keeping filenames unchanged."""
    split_map = {
        "bounding_box_train": "bounding_box_train",
        "query": "query",
        "bounding_box_test": "bounding_box_test",
    }
    for src_split, dst_split in split_map.items():
        src_dir = src / src_split
        dst_dir = dst / dst_split
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = list(src_dir.glob("*.jpg"))
        for f in tqdm(files, desc=f"Market-1501 {src_split}", unit="img"):
            if _MARKET_RE.match(f.name):
                shutil.copy2(f, dst_dir / f.name)


# ── MCC_pedestrian ────────────────────────────────────────────────────────────

def _copy_mcc(src: Path, dst: Path) -> None:
    """Copy MCC_pedestrian splits to dst, converting filenames to Market-1501 format."""
    split_map = {
        "bounding_box_train": "bounding_box_train",
        "query": "query",
        "bounding_box_test": "bounding_box_test",
    }
    for src_split, dst_split in split_map.items():
        src_dir = src / src_split
        dst_dir = dst / dst_split
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = list(src_dir.glob("*.jpg"))
        for f in tqdm(files, desc=f"MCC {src_split}", unit="img"):
            m = _MCC_RE.match(f.name)
            if m is None:
                continue
            pid = int(m.group(1)) + _MCC_PID_OFFSET
            cam = int(m.group(2))
            frame = int(m.group(3)) % 1_000_000   # clamp to 6 digits
            new_name = f"{pid:04d}_c{cam}s1_{frame:06d}_00.jpg"
            shutil.copy2(f, dst_dir / new_name)


# ── Wildtrack ─────────────────────────────────────────────────────────────────

def _build_wildtrack_pid_map(ann_dir: Path) -> dict[int, int]:
    """Return {raw_personID → remapped_pid} using sequential 1-indexed mapping."""
    raw_pids: set[int] = set()
    for ann_file in ann_dir.glob("*.json"):
        for entry in json.loads(ann_file.read_text()):
            raw_pids.add(entry["personID"])
    return {raw: _WILDTRACK_PID_OFFSET + i + 1 for i, raw in enumerate(sorted(raw_pids))}


def _extract_wildtrack(src: Path, dst: Path) -> None:
    """Crop person bboxes from Wildtrack scene images → dst/bounding_box_train."""
    ann_dir = src / "annotations_positions"
    img_root = src / "Image_subsets"
    out_dir = dst / "bounding_box_train"
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_map = _build_wildtrack_pid_map(ann_dir)
    ann_files = sorted(ann_dir.glob("*.json"))

    for ann_file in tqdm(ann_files, desc="Wildtrack crops", unit="frame"):
        # Frame number from filename, e.g. "00000025.json" → 25
        frame_num = int(ann_file.stem)
        frame_idx = frame_num % 1_000_000

        annotations = json.loads(ann_file.read_text())
        for entry in annotations:
            pid = pid_map[entry["personID"]]
            for view in entry["views"]:
                view_num = view["viewNum"]   # 0-indexed
                xmin, xmax = view["xmin"], view["xmax"]
                ymin, ymax = view["ymin"], view["ymax"]

                # Skip views where the person is not visible
                if xmin < 0 or ymin < 0:
                    continue

                cam = view_num + 1   # 1-indexed camera ID
                img_path = img_root / f"C{cam}" / f"{frame_num:08d}.png"
                if not img_path.exists():
                    continue

                img = Image.open(img_path).convert("RGB")
                w, h = img.size

                # Clamp bbox to image bounds
                x0 = max(0, min(xmin, w - 1))
                y0 = max(0, min(ymin, h - 1))
                x1 = max(0, min(xmax, w))
                y1 = max(0, min(ymax, h))

                if (x1 - x0) < _MIN_CROP_PX or (y1 - y0) < _MIN_CROP_PX:
                    continue

                crop = img.crop((x0, y0, x1, y1))
                det = view_num   # reuse viewNum as det index (0–6, fits 2 digits)
                out_name = f"{pid:04d}_c{cam}s1_{frame_idx:06d}_{det:02d}.jpg"
                crop.save(out_dir / out_name, quality=95)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True, help="Root of pedestrian data directory")
    parser.add_argument("--dst", type=Path, required=True, help="Output Dataset directory")
    parser.add_argument("--skip-market", action="store_true", help="Skip Market-1501 copy")
    parser.add_argument("--skip-mcc", action="store_true", help="Skip MCC_pedestrian copy")
    parser.add_argument("--skip-wildtrack", action="store_true", help="Skip Wildtrack crop extraction")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    src: Path = args.src
    dst: Path = args.dst

    market_src = src / "Market-1501-v15.09.15"
    mcc_src = src / "MCC_pedestrian"
    wildtrack_src = src / "Wildtrack"

    for name, path in [("Market-1501", market_src), ("MCC_pedestrian", mcc_src), ("Wildtrack", wildtrack_src)]:
        if not path.exists():
            print(f"[WARN] {name} not found at {path} — skipping")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"Output: {dst}")

    if not args.skip_market and market_src.exists():
        print("\n=== Market-1501 ===")
        _copy_market(market_src, dst)

    if not args.skip_mcc and mcc_src.exists():
        print("\n=== MCC_pedestrian ===")
        _copy_mcc(mcc_src, dst)

    if not args.skip_wildtrack and wildtrack_src.exists():
        print("\n=== Wildtrack ===")
        _extract_wildtrack(wildtrack_src, dst)

    print("\n=== Summary ===")
    for split in ("bounding_box_train", "query", "bounding_box_test"):
        split_dir = dst / split
        count = len(list(split_dir.glob("*.jpg"))) if split_dir.exists() else 0
        print(f"  {split:<22}: {count:>7} images")


if __name__ == "__main__":
    sys.exit(main())
