"""Stage 1 training: optimize PromptLearner with frozen encoders.

Loss: SupConLoss(img→text) + SupConLoss(text→img)
Trainable: prompt_learner only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.defaults import PedestrianReIDConfig
from src.datasets.pedestrian_dataset import build_pedestrian_loaders
from src.losses.sup_con_loss import SupConLoss
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


def train_stage1(cfg: PedestrianReIDConfig) -> Path:
    """Run Stage 1 training and return path to saved checkpoint."""
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.fp16 else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"[Stage 1] device={device}  fp16={cfg.fp16}")
        print(f"[Stage 1] market1501_root={cfg.market1501_root}")
        print(f"[Stage 1] mini={cfg.mini}  epochs={cfg.effective_epochs_stage1}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, _, _, num_pids, num_cams = build_pedestrian_loaders(
        market1501_root=cfg.market1501_root,
        batch_size=cfg.batch_size,
        num_instances=cfg.num_instances,
        num_workers=cfg.num_workers,
        mini=cfg.mini,
        mini_num_ids=cfg.mini_num_ids,
    )
    if accelerator.is_main_process:
        print(f"[Stage 1] train_pids={num_pids}  train_cams={num_cams}  images={len(train_loader.dataset)}  batches={len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CLIPReIDPedestrianModel(
        num_pids=num_pids,
        num_cams=num_cams,
        num_views=cfg.num_views,
        olp_k=cfg.olp_k,
        clip_name=cfg.clip_model_name,
        template=cfg.prompt_template,
        n_ctx=cfg.n_ctx,
    )
    model.freeze_for_stage1()

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Stage 1] trainable={trainable:,}  total={total:,}")

    # ── Optimiser + LR schedule ───────────────────────────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_stage1,
        weight_decay=cfg.weight_decay,
    )
    warmup = LinearLR(optimizer, start_factor=1e-5 / cfg.lr_stage1, end_factor=1.0, total_iters=cfg.warmup_epochs)
    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.effective_epochs_stage1 - cfg.warmup_epochs)
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[cfg.warmup_epochs])

    criterion = SupConLoss(temperature=1.0)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(1, cfg.effective_epochs_stage1 + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (imgs, pids, cam_ids, view_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            out = model(imgs, pids, cam_ids, view_ids)
            img_feat = out["img_feat"]         # (N, 512)
            text_feat = out["text_feat"]       # (U, 512), U = unique pids
            batch_pids = out["batch_pids"]     # (U,)

            loss_i2t = criterion(img_feat, text_feat, pids, batch_pids)
            loss_t2i = criterion(text_feat, img_feat, batch_pids, pids)
            loss = loss_i2t + loss_t2i

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
                )
            optimizer.step()
            epoch_loss += loss.item()

            if accelerator.is_main_process and batch_idx % 20 == 0:
                print(
                    f"  Epoch {epoch:3d}/{cfg.effective_epochs_stage1}"
                    f"  batch {batch_idx:4d}/{len(train_loader)}"
                    f"  loss={loss.item():.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        if accelerator.is_main_process:
            print(f"[Stage 1] epoch={epoch}  avg_loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Periodic checkpoint
        if epoch % cfg.checkpoint_period == 0 or epoch == cfg.effective_epochs_stage1:
            ckpt_path = cfg.output_dir / f"stage1_epoch{epoch:03d}.pth"
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": 1,
                        "model_state": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "num_pids": num_pids,
                        "num_cams": num_cams,
                    },
                    ckpt_path,
                )
                print(f"[Stage 1] saved → {ckpt_path}")

    final = cfg.output_dir / f"stage1_epoch{cfg.effective_epochs_stage1:03d}.pth"
    if accelerator.is_main_process:
        print(f"[Stage 1] complete. best_loss={best_loss:.4f}")
    return final


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP-ReID Stage 1: PromptLearner training")
    p.add_argument("--market1501-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mini", action="store_true", help="Run on 1000-image mini dataset")
    p.add_argument("--mini-num-ids", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
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

    train_stage1(cfg)
