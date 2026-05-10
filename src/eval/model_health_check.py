"""Model health check — 8 PASS/FAIL checks run before LoRA fine-tuning."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from src.config.defaults import PedestrianReIDConfig


def run_health_checks(
    model,
    cfg: PedestrianReIDConfig,
    sample_batch: tuple | None = None,
    device: torch.device | None = None,
) -> bool:
    """Run all health checks and print results.

    Args:
        model:        Trained CLIPReIDPedestrianModel.
        cfg:          Pipeline config.
        sample_batch: Optional (imgs, pids, cam_ids, view_ids) for forward checks.
        device:       Target device; defaults to model parameter device.

    Returns:
        True if all checks pass, False otherwise.
    """
    if device is None:
        device = next(model.parameters()).device

    results: list[tuple[str, bool, str]] = []

    # Check 1 & 2: prompt template
    results.append((
        "prompt_template contains 'person'",
        "person" in cfg.prompt_template.lower(),
        "",
    ))
    results.append((
        "prompt_template has no 'vehicle'",
        "vehicle" not in cfg.prompt_template.lower(),
        "",
    ))

    if sample_batch is not None:
        imgs, pids, cam_ids, view_ids = [x.to(device) for x in sample_batch]

        model.eval()
        with torch.no_grad():
            # FP32 forward
            out32 = model(imgs.float(), pids, cam_ids, view_ids)
            feat32 = out32["fused_feat"]

            # FP16 forward (only on CUDA)
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out16 = model(imgs, pids, cam_ids, view_ids)
                feat16 = out16["fused_feat"]
            else:
                feat16 = feat32

        # Check 3: NaN/Inf
        results.append((
            "No NaN/Inf in fused_feat",
            not (torch.isnan(feat32).any() or torch.isinf(feat32).any()),
            "",
        ))

        # Check 4: feature norms in [0.1, 2.0] (L2-normalised expected ≈1)
        norms = feat32.norm(dim=1)
        results.append((
            "Feature norms in [0.1, 2.0]",
            bool((norms >= 0.1).all() and (norms <= 2.0).all()),
            f"min={norms.min():.3f} max={norms.max():.3f}",
        ))

        # Check 5: SIE cam embeddings have non-zero weights
        cam_w = model.sie.cam_embedding.weight
        results.append((
            "SIE cam_embedding weights non-zero",
            bool(cam_w.abs().sum() > 0),
            "",
        ))

        # Check 6: OLP selects > 0 patches
        results.append((
            "OLP selected > 0 patches",
            model.olp.last_selected_k > 0,
            f"k={model.olp.last_selected_k}",
        ))

        # Check 7: FP16 vs FP32 similarity
        if device.type == "cuda":
            diff = (feat16.float() - feat32).abs().max().item()
            results.append((
                "FP16 vs FP32 diff < 0.05",
                diff < 0.05,
                f"max_diff={diff:.4f}",
            ))

        # Check 8: classification scores non-constant (model not collapsed)
        cls_std = out32["cls_score"].std().item()
        results.append((
            "cls_score std > 1e-4 (not collapsed)",
            cls_std > 1e-4,
            f"std={cls_std:.4f}",
        ))

    # Print results
    all_pass = True
    print("\n── Model Health Check ──────────────────────────────")
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not passed:
            all_pass = False
    print("────────────────────────────────────────────────────")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'FAILED ✗'}\n")

    return all_pass


if __name__ == "__main__":
    import argparse
    from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-pids", type=int, default=751)
    parser.add_argument("--num-cams", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPReIDPedestrianModel(num_pids=args.num_pids, num_cams=args.num_cams).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    cfg = PedestrianReIDConfig()
    ok = run_health_checks(model, cfg, device=device)
    sys.exit(0 if ok else 1)
