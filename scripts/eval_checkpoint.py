"""Evaluate a Stage 2 checkpoint with configurable post-processing.

Usage:
    .venv/bin/python scripts/eval_checkpoint.py \
        --checkpoint model/custom/stage2_best.pth

    # Disable specific post-processing for ablation
    .venv/bin/python scripts/eval_checkpoint.py \
        --checkpoint model/custom/stage2_best.pth \
        --no-flip-tta --no-qe --no-power-norm

    # Custom reranking params
    .venv/bin/python scripts/eval_checkpoint.py \
        --checkpoint model/custom/stage2_best.pth \
        --rerank-lambda 0.5 --rerank-k1 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.defaults import PedestrianReIDConfig
from src.datasets.pedestrian_dataset import build_pedestrian_loaders
from src.eval.evaluate import evaluate
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


def main() -> None:
    args = _parse_args()
    cfg = PedestrianReIDConfig()
    if args.market1501_root:
        cfg.market1501_root = args.market1501_root

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[eval] device={device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_pids = ckpt.get("num_pids", 751)
    num_cams = ckpt.get("num_cams", cfg.num_cams)
    epoch = ckpt.get("epoch", "?")
    saved_map = ckpt.get("best_map", "?")
    print(f"[eval] checkpoint={ckpt_path}  epoch={epoch}  saved_mAP={saved_map}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = CLIPReIDPedestrianModel(
        num_pids=num_pids,
        num_cams=num_cams,
        num_views=cfg.num_views,
        olp_k=cfg.olp_k,
        sie_coe=cfg.sie_coe,
        clip_name=cfg.clip_model_name,
        template=cfg.prompt_template,
        n_ctx=cfg.n_ctx,
    ).to(device)

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[eval] model loaded: num_pids={num_pids}  num_cams={num_cams}")

    # ── Data ─────────────────────────────────────────────────────────────────
    _, query_loader, gallery_loader, _, _ = build_pedestrian_loaders(
        market1501_root=cfg.market1501_root,
        batch_size=args.batch_size,
        num_instances=cfg.num_instances,
        num_workers=args.num_workers,
    )

    # ── Post-processing config ───────────────────────────────────────────────
    flip_tta = not args.no_flip_tta
    power_alpha = 0.0 if args.no_power_norm else args.power_norm_alpha
    qe_k = 0 if args.no_qe else args.qe_k
    rerank_lambda = args.rerank_lambda

    print(f"[eval] flip_tta={flip_tta}  power_norm={power_alpha}  "
          f"qe_k={qe_k}  qe_alpha={args.qe_alpha}  "
          f"rerank(k1={args.rerank_k1}, k2={args.rerank_k2}, λ={rerank_lambda})")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    use_fp16 = device.type == "cuda"
    metrics = evaluate(
        model, query_loader, gallery_loader, device,
        fp16=use_fp16,
        use_rerank=not args.no_rerank,
        k1=args.rerank_k1,
        k2=args.rerank_k2,
        lambda_=rerank_lambda,
        flip_tta=flip_tta,
        power_norm_alpha=power_alpha,
        qe_k=qe_k,
        qe_alpha=args.qe_alpha,
    )

    print(f"\n{'='*50}")
    print(f"  mAP   = {metrics['mAP']:.2f}%")
    print(f"  Rank-1 = {metrics['rank1']:.2f}%")
    print(f"  Rank-5 = {metrics['rank5']:.2f}%")
    print(f"  Rank-10= {metrics['rank10']:.2f}%")
    print(f"{'='*50}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Stage 2 Re-ID checkpoint")
    p.add_argument("--checkpoint", "--ckpt", type=Path, required=True,
                   help="Path to stage2 .pth checkpoint")
    p.add_argument("--market1501-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    # Post-processing toggles
    p.add_argument("--no-flip-tta", action="store_true", help="Disable horizontal flip TTA")
    p.add_argument("--no-power-norm", action="store_true", help="Disable power normalization")
    p.add_argument("--no-qe", action="store_true", help="Disable alpha-query expansion")
    p.add_argument("--no-rerank", action="store_true", help="Disable k-reciprocal re-ranking")

    # Post-processing params
    p.add_argument("--power-norm-alpha", type=float, default=0.5)
    p.add_argument("--qe-k", type=int, default=5)
    p.add_argument("--qe-alpha", type=float, default=3.0)
    p.add_argument("--rerank-k1", type=int, default=20)
    p.add_argument("--rerank-k2", type=int, default=6)
    p.add_argument("--rerank-lambda", type=float, default=0.7)

    return p.parse_args()


if __name__ == "__main__":
    main()
