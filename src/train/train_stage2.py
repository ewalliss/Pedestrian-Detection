"""Stage 2 training: fine-tune image encoder using frozen text features.

Loss: 0.25 * (ID_global + ID_proj) + 1.0 * (Circle_global + Circle_fused)
      + 1.0 * I2T_SupCon
Schedule: LinearLR warmup (10 epochs) → CosineAnnealingLR
Trainable: image_encoder + SIE + OLP + BN_global + BN_proj + classifiers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.defaults import PedestrianReIDConfig
from src.datasets.pedestrian_dataset import build_pedestrian_loaders
from src.eval.evaluate import evaluate
from src.eval.model_health_check import run_health_checks
from src.losses.circle_loss import CircleLoss
from src.losses.id_loss import IDLoss
from src.losses.sup_con_loss import SupConLoss
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


def train_stage2(cfg: PedestrianReIDConfig, stage1_ckpt: Path, resume_ckpt: Path | None = None) -> Path:
    """Run Stage 2 training and return path to best checkpoint."""
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.fp16 else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, query_loader, gallery_loader, num_pids, num_cams = build_pedestrian_loaders(
        market1501_root=cfg.market1501_root,
        batch_size=cfg.batch_size,
        num_instances=cfg.num_instances,
        num_workers=cfg.num_workers,
        mini=cfg.mini,
        mini_num_ids=cfg.mini_num_ids,
    )
    if accelerator.is_main_process:
        print(f"[Stage 2] train_pids={num_pids}  images={len(train_loader.dataset)}  batches={len(train_loader)}")

    # ── Model: load Stage 1 checkpoint ────────────────────────────────────────
    ckpt = torch.load(stage1_ckpt, map_location=device)
    # Always build model with CURRENT data's num_pids/num_cams so the
    # classifier matches the training set (handles mini→full transitions).
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
    # Load only weights whose shapes match (handles mini→full num_pids change).
    # class_ctx and classifier resize with num_pids — skip and re-init those.
    ckpt_state = ckpt["model_state"]
    model_state = model.state_dict()
    compatible = {k: v for k, v in ckpt_state.items()
                  if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in ckpt_state if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    if skipped and accelerator.is_main_process:
        print(f"[Stage 2] re-initialised {len(skipped)} size-mismatched tensors: {skipped}")
    model.freeze_for_stage2()
    if accelerator.is_main_process:
        print(f"[Stage 2] loaded Stage 1 from {stage1_ckpt}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Stage 2] trainable={trainable:,}")

    # ── Precompute text features once (text path is frozen) ───────────────────
    # Must be done BEFORE accelerator.prepare() to avoid DDP wrapping issues.
    # .detach() prevents gradient flow through the frozen text encoder (C-2 fix).
    model.eval()
    with torch.no_grad():
        all_pids = torch.arange(model.num_pids, device=device)
        text_feats = model.encode_text(all_pids).detach()  # (num_pids, 512)
    if accelerator.is_main_process:
        print(f"[Stage 2] precomputed text_feats: {text_feats.shape}")

    # ── Losses ────────────────────────────────────────────────────────────────
    id_loss_fn = IDLoss(num_classes=model.num_pids, epsilon=cfg.label_smoothing).to(device)
    circle_fn = CircleLoss(gamma=cfg.circle_gamma, margin=cfg.circle_margin).to(device)
    sup_con_fn = SupConLoss(temperature=cfg.i2t_temperature).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_stage2,
        weight_decay=cfg.weight_decay,
    )
    # CosineAnnealingLR with optional linear warmup
    total_epochs = cfg.effective_epochs_stage2
    warmup_epochs = min(cfg.lr_warmup_epochs_stage2, total_epochs)
    if warmup_epochs > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=cfg.lr_min_stage2)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=cfg.lr_min_stage2)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    best_map = 0.0
    best_path = cfg.output_dir / "stage2_best.pth"
    no_improve = 0
    start_epoch = 1

    # ── Resume from Stage 2 checkpoint ─────────────────────────────────────────
    if resume_ckpt is not None:
        resume = torch.load(resume_ckpt, map_location=device, weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(resume["model_state"])
        if "optimizer_state" in resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = resume.get("epoch", 0) + 1
        best_map = resume.get("best_map", 0.0)
        if accelerator.is_main_process:
            print(f"[Stage 2] resumed from epoch {start_epoch - 1}  best_mAP={best_map:.2f}%")

    for epoch in range(start_epoch, cfg.effective_epochs_stage2 + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, pids, cam_ids, view_ids in train_loader:
            optimizer.zero_grad()

            out = model(imgs, pids, cam_ids, view_ids)
            global_feat = out["global_feat"]       # (N, 512)
            fused_feat = out["fused_feat"]         # (N, 512)
            cls_score = out["cls_score"]           # (N, num_pids) from 768-dim
            cls_score_proj = out["cls_score_proj"] # (N, num_pids) from 512-dim

            # Dual-head ID loss (applied to both classifier heads)
            loss_id = id_loss_fn(cls_score, pids) + id_loss_fn(cls_score_proj, pids)

            # Multi-stream Circle Loss (self-paced metric learning on both streams)
            loss_circle = circle_fn(global_feat, pids) + circle_fn(fused_feat, pids)

            # I2T: align image features to precomputed text features
            txt = text_feats[pids]                                   # (N, 512)
            loss_i2t = sup_con_fn(fused_feat, txt, pids, pids)

            loss = (
                cfg.id_loss_weight * loss_id
                + cfg.circle_loss_weight * loss_circle
                + cfg.i2t_loss_weight * loss_i2t
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        if accelerator.is_main_process:
            print(f"[Stage 2] epoch={epoch}  loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # Evaluate every 5 epochs (or every epoch in mini mode)
        eval_freq = 1 if cfg.mini else 5
        if epoch % eval_freq == 0 or epoch == cfg.effective_epochs_stage2:
            metrics = evaluate(
                accelerator.unwrap_model(model),
                query_loader, gallery_loader, device,
                fp16=accelerator.mixed_precision == "fp16",
                use_rerank=True,
                k1=cfg.rerank_k1,
                k2=cfg.rerank_k2,
                lambda_=cfg.rerank_lambda,
                flip_tta=cfg.flip_tta,
                power_norm_alpha=cfg.power_norm_alpha,
                qe_k=cfg.qe_k,
                qe_alpha=cfg.qe_alpha,
            )
            map_val = metrics["mAP"]
            if accelerator.is_main_process:
                print(f"[Stage 2] epoch={epoch}  mAP={map_val:.2f}%  R1={metrics['rank1']:.2f}%")

            if map_val > best_map:
                best_map = map_val
                no_improve = 0
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(
                        {
                            "epoch": epoch,
                            "stage": 2,
                            "model_state": accelerator.unwrap_model(model).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "num_pids": accelerator.unwrap_model(model).num_pids,
                            "num_cams": num_cams,
                            "best_map": best_map,
                        },
                        best_path,
                    )
                    print(f"[Stage 2] ✓ new best mAP={best_map:.2f}% → {best_path}")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience and not cfg.mini:
                    if accelerator.is_main_process:
                        print(f"[Stage 2] early stop (no improve for {no_improve} evals)")
                    break

    if accelerator.is_main_process:
        print(f"[Stage 2] complete. best_mAP={best_map:.2f}%")

    # Run health checks before declaring done
    sample_batch = next(iter(train_loader))
    sample_batch = tuple(x[:8] for x in sample_batch)  # small batch for checks
    run_health_checks(accelerator.unwrap_model(model), cfg, sample_batch, device)

    return best_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP-ReID Stage 2: image encoder fine-tuning")
    p.add_argument("--stage1-checkpoint", "--ckpt", dest="stage1_checkpoint", type=Path, required=True)
    p.add_argument("--market1501-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mini", action="store_true")
    p.add_argument("--mini-num-ids", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume Stage 2 from an existing Stage 2 checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = PedestrianReIDConfig()
    if args.market1501_root:
        cfg.market1501_root = args.market1501_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.mini = args.mini
    cfg.mini_num_ids = args.mini_num_ids
    cfg.num_workers = args.num_workers
    cfg.fp16 = not args.no_fp16

    train_stage2(cfg, args.stage1_checkpoint, resume_ckpt=args.resume)
