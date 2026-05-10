"""LoRA fine-tuning for CLIPReIDPedestrianModel using PEFT.

Applies LoRA adapters to the ViT image encoder's q_proj and v_proj layers.
Freezes all non-LoRA parameters.  Trains with the same Stage 2 loss:
  0.25 * ID_CE + 1.0 * Triplet + 1.0 * I2T_SupCon

LoRA config: r=16, alpha=16, dropout=0.1, target=["q_proj","v_proj"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.defaults import PedestrianReIDConfig
from src.datasets.pedestrian_dataset import build_pedestrian_loaders
from src.eval.evaluate import evaluate
from src.losses.id_loss import IDLoss
from src.losses.sup_con_loss import SupConLoss
from src.losses.triplet_loss import TripletLoss
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


def _apply_lora(model: CLIPReIDPedestrianModel, r: int, lora_alpha: int, lora_dropout: float) -> None:
    """Inject LoRA adapters into q_proj and v_proj of the ViT encoder."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise RuntimeError(
            "PEFT not installed. Run: pip install peft"
        )

    # LoRA targets: q_proj and v_proj in the ViT transformer blocks
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    # Wrap only the visual transformer with LoRA
    model.image_encoder = get_peft_model(model.image_encoder, lora_cfg)

    # Re-freeze everything except LoRA params + classifier head + SIE + OLP + BN
    for name, param in model.named_parameters():
        if "lora_" not in name:
            # Keep stage2 trainable components unfrozen
            keep = any(k in name for k in ["sie.", "olp.", "classifier.", "bn.", "visual_proj"])
            param.requires_grad = keep


def lora_finetune(cfg: PedestrianReIDConfig, stage2_ckpt: Path) -> Path:
    """Run LoRA fine-tuning and return path to best checkpoint."""
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.fp16 else "no",
        gradient_accumulation_steps=cfg.lora_accum_steps,
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
        print(f"[LoRA] train_pids={num_pids}  batches={len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(stage2_ckpt, map_location=device)
    model = CLIPReIDPedestrianModel(
        num_pids=ckpt.get("num_pids", num_pids),
        num_cams=ckpt.get("num_cams", num_cams),
        num_views=cfg.num_views,
        olp_k=cfg.olp_k,
        clip_name=cfg.clip_model_name,
        template=cfg.prompt_template,
        n_ctx=cfg.n_ctx,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    if accelerator.is_main_process:
        print(f"[LoRA] loaded Stage 2 from {stage2_ckpt}")

    _apply_lora(model, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] trainable={trainable:,}  total={total:,}  ({100*trainable/total:.1f}%)")

    # ── Precompute text features (frozen path; .detach() is C-2 fix) ──────────
    # Done BEFORE accelerator.prepare() to avoid DDP wrapping issues.
    model.eval()
    with torch.no_grad():
        all_pids = torch.arange(model.num_pids, device=device)
        text_feats = model.encode_text(all_pids).detach()

    # ── Losses ────────────────────────────────────────────────────────────────
    id_loss_fn = IDLoss(num_classes=model.num_pids, epsilon=cfg.label_smoothing).to(device)
    triplet_fn = TripletLoss(margin=cfg.triplet_margin).to(device)
    sup_con_fn = SupConLoss(temperature=1.0).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lora_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.lora_epochs)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    accelerator.register_for_checkpointing(train_loader.batch_sampler)

    best_map = 0.0
    best_path = cfg.output_dir / "lora_best.pth"

    for epoch in range(1, cfg.lora_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, pids, cam_ids, view_ids in train_loader:
            with accelerator.accumulate(model):
                out = model(imgs, pids, cam_ids, view_ids)
                fused = out["fused_feat"]
                cls_score = out["cls_score"]

                txt = text_feats[pids]
                loss_i2t = sup_con_fn(fused, txt, pids, pids)
                loss_id = id_loss_fn(cls_score, pids)
                loss_tri = triplet_fn(fused, pids)

                loss = (
                    cfg.id_loss_weight * loss_id
                    + cfg.triplet_loss_weight * loss_tri
                    + cfg.i2t_loss_weight * loss_i2t
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 1.0
                    )
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        if accelerator.is_main_process:
            print(f"[LoRA] epoch={epoch}  loss={avg_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        eval_freq = 1 if cfg.mini else 5
        if epoch % eval_freq == 0 or epoch == cfg.lora_epochs:
            metrics = evaluate(
                accelerator.unwrap_model(model),
                query_loader, gallery_loader, device,
                fp16=accelerator.mixed_precision == "fp16",
                use_rerank=False,
            )
            map_val = metrics["mAP"]
            if accelerator.is_main_process:
                print(f"[LoRA] epoch={epoch}  mAP={map_val:.2f}%  R1={metrics['rank1']:.2f}%")

            if map_val > best_map:
                best_map = map_val
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(
                        {
                            "epoch": epoch,
                            "stage": "lora",
                            "model_state": accelerator.unwrap_model(model).state_dict(),
                            "num_pids": accelerator.unwrap_model(model).num_pids,
                            "num_cams": num_cams,
                            "best_map": best_map,
                        },
                        best_path,
                    )
                    print(f"[LoRA] ✓ new best mAP={best_map:.2f}% → {best_path}")

    if accelerator.is_main_process:
        print(f"[LoRA] complete. best_mAP={best_map:.2f}%")
    return best_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for CLIP-ReID pedestrian model")
    p.add_argument("--stage2-checkpoint", type=Path, required=True)
    p.add_argument("--market1501-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mini", action="store_true")
    p.add_argument("--mini-num-ids", type=int, default=100)
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
    cfg.mini = args.mini
    cfg.mini_num_ids = args.mini_num_ids
    cfg.num_workers = args.num_workers
    cfg.fp16 = not args.no_fp16

    lora_finetune(cfg, args.stage2_checkpoint)
