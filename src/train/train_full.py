"""Full two-stage training: Stage 1 (PromptLearner) → Stage 2 (image encoder).

Runs both stages sequentially, passing the Stage 1 checkpoint automatically
into Stage 2. All hyperparameters come from PedestrianReIDConfig.

Usage:
    # Full run
    python -m src.train.train_full --market1501-root /path/to/Market-1501-v15.09.15

    # Smoke test (3 + 2 epochs, 100 IDs)
    python -m src.train.train_full --mini

    # Resume from existing Stage 1 checkpoint (skip Stage 1)
    python -m src.train.train_full --skip-stage1 --stage1-checkpoint output/pedestrian/stage1_epoch120.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.defaults import PedestrianReIDConfig
from src.train.train_stage1 import train_stage1
from src.train.train_stage2 import train_stage2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP-ReID full two-stage training")
    p.add_argument("--market1501-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mini", action="store_true", help="Smoke test: 3+2 epochs, 100 IDs")
    p.add_argument("--mini-num-ids", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 and use existing checkpoint")
    p.add_argument("--stage1-checkpoint", type=Path, default=None,
                   help="Path to Stage 1 checkpoint (required when --skip-stage1)")
    p.add_argument("--resume-stage2", type=Path, default=None,
                   help="Resume Stage 2 from an existing Stage 2 checkpoint")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = PedestrianReIDConfig()
    if args.market1501_root:
        cfg.market1501_root = args.market1501_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.batch_size:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.mini = args.mini
    cfg.mini_num_ids = args.mini_num_ids
    cfg.fp16 = not args.no_fp16

    print("=" * 60)
    print("V-Track  —  Full Two-Stage Training")
    print(f"  Stage 1 epochs : {cfg.effective_epochs_stage1}")
    print(f"  Stage 2 epochs : {cfg.effective_epochs_stage2}")
    print(f"  Data root      : {cfg.market1501_root}")
    print(f"  Output dir     : {cfg.output_dir}")
    print(f"  fp16           : {cfg.fp16}  |  mini: {cfg.mini}")
    print("=" * 60)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.skip_stage1:
        if args.stage1_checkpoint is None:
            print("[ERROR] --stage1-checkpoint is required when using --skip-stage1")
            sys.exit(1)
        stage1_ckpt = args.stage1_checkpoint
        print(f"\n[Stage 1] skipped — using checkpoint: {stage1_ckpt}")
    else:
        print("\n" + "─" * 60)
        print("STAGE 1: PromptLearner training")
        print("─" * 60)
        stage1_ckpt = train_stage1(cfg)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STAGE 2: Image encoder fine-tuning")
    print("─" * 60)
    stage2_ckpt = train_stage2(cfg, stage1_ckpt, resume_ckpt=args.resume_stage2)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Stage 1 checkpoint : {stage1_ckpt}")
    print(f"  Stage 2 best model : {stage2_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
