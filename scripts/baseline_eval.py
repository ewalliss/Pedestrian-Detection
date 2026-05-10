"""Baseline vs Custom comparison — original CLIP-ReID ViT-B/16 on Market-1501.

Evaluates both checkpoints and prints a side-by-side comparison table.
Also saves baseline metrics to JSON (consumed by run_experiment.py --skip-baseline).

Usage
-----
python scripts/baseline_eval.py \
    --market1501-root /path/to/Market-1501-v15.09.15

    # Skip reranking (faster)
python scripts/baseline_eval.py \
    --market1501-root /path/to/Market-1501-v15.09.15 \
    --no-rerank

Flags
-----
--no-rerank       Skip k-reciprocal re-ranking.
--batch-size N    DataLoader batch size (default 128).
--num-workers N   DataLoader workers (default 2).
--no-fp16         Disable FP16 for custom model (use on CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
_ORIG_REID = _REPO / "original_CLIP-REID" / "CLIP-ReID"
_BASELINE_CKPT = (
    _REPO / "model" / "model_baseline" / "Market1501_clipreid_12x12sie_ViT-B-16_60.pth"
)
_CUSTOM_CKPT = _REPO / "model" / "custom" / "stage2_best.pth"
_BASELINE_JSON = _REPO / "model" / "model_baseline" / "baseline_metrics.json"

sys.path.insert(0, str(_REPO))

from src.datasets.market1501 import Market1501
from src.datasets.transforms import build_val_transform
from src.eval.evaluate import compute_metrics
from src.eval.reranking import k_reciprocal_rerank
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


@dataclass
class EvalResult:
    model_name: str
    mAP: float
    rank1: float
    rank5: float
    rank10: float
    mAP_rr: float
    rank1_rr: float
    num_params_m: float
    inference_ms: float


# ── Model loading ──────────────────────────────────────────────────────────────


def _load_baseline(device: torch.device):
    """Build original CLIP-ReID ViT-B/16 (stride-12, SIE-cam) and load checkpoint.

    Args:
        device: Target device.

    Returns:
        Model in eval mode.
    """
    sys.path.insert(0, str(_ORIG_REID))
    try:
        from config import cfg as orig_cfg
        from model.make_model_clipreid import make_model as orig_make_model
    finally:
        sys.path.pop(0)

    orig_cfg.defrost()
    orig_cfg.MODEL.NAME = "ViT-B-16"
    orig_cfg.MODEL.STRIDE_SIZE = [12, 12]
    orig_cfg.MODEL.SIE_CAMERA = True
    orig_cfg.MODEL.SIE_VIEW = False
    orig_cfg.MODEL.SIE_COE = 3.0
    orig_cfg.MODEL.NECK = "bnneck"
    orig_cfg.MODEL.COS_LAYER = False
    orig_cfg.INPUT.SIZE_TRAIN = [256, 128]
    orig_cfg.INPUT.SIZE_TEST = [256, 128]
    orig_cfg.TEST.NECK_FEAT = "before"
    orig_cfg.TEST.FEAT_NORM = "yes"
    orig_cfg.freeze()

    model = orig_make_model(orig_cfg, num_class=751, camera_num=6, view_num=1)
    model.load_param(str(_BASELINE_CKPT))
    return model.to(device).eval()


def _load_custom(device: torch.device) -> tuple[CLIPReIDPedestrianModel, float]:
    """Load fine-tuned CLIPReIDPedestrianModel from stage2_best.pth.

    Args:
        device: Target device.

    Returns:
        (model in eval mode, parameter count in millions)
    """
    ckpt = torch.load(_CUSTOM_CKPT, map_location="cpu", weights_only=False)
    num_pids = ckpt.get("num_pids", 751)
    num_cams = ckpt.get("num_cams", 30)
    print(f"  Checkpoint metadata: num_pids={num_pids}  num_cams={num_cams}"
          f"  best_mAP={ckpt.get('best_map', '?')}")
    model = CLIPReIDPedestrianModel(num_pids=num_pids, num_cams=num_cams)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return model.to(device).eval(), n_params


# ── Feature extraction ─────────────────────────────────────────────────────────


@torch.no_grad()
def _extract_baseline(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract L2-normalised features from the baseline model.

    Args:
        model:  Original CLIP-ReID model.
        loader: DataLoader yielding (imgs, pids, cam_ids, view_ids).
        device: Computation device.

    Returns:
        feats (N, D), pids (N,), cam_ids (N,)
    """
    feats, pids_out, cams_out = [], [], []
    for imgs, pids, cam_ids, _view_ids in tqdm(loader, desc="  extracting", leave=False):
        imgs = imgs.to(device)
        # Market-1501 cam_ids are 1-indexed (1–6); original model expects 0-indexed
        cam_label = (cam_ids - 1).to(device)
        feat = model(imgs, cam_label=cam_label)  # (N, 1280) in eval mode
        feats.append(F.normalize(feat.float(), p=2, dim=-1).cpu())
        pids_out.append(pids)
        cams_out.append(cam_ids.cpu())
    return torch.cat(feats), torch.cat(pids_out), torch.cat(cams_out)


@torch.no_grad()
def _extract_custom(
    model: CLIPReIDPedestrianModel,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract L2-normalised features from the custom model.

    Args:
        model:  CLIPReIDPedestrianModel.
        loader: DataLoader yielding (imgs, pids, cam_ids, view_ids).
        device: Computation device.
        fp16:   Whether to use FP16 autocast (CUDA only).

    Returns:
        feats (N, D), pids (N,), cam_ids (N,)
    """
    feats, pids_out, cams_out = [], [], []
    use_fp16 = fp16 and device.type == "cuda"
    for imgs, pids, cam_ids, view_ids in tqdm(loader, desc="  extracting", leave=False):
        imgs = imgs.to(device)
        cam_ids_d = cam_ids.to(device)
        view_ids_d = view_ids.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            feat = model.extract_features(imgs, cam_ids_d, view_ids_d)  # (N, 512)
        feats.append(feat.float().cpu())
        pids_out.append(pids)
        cams_out.append(cam_ids.cpu())
    return torch.cat(feats), torch.cat(pids_out), torch.cat(cams_out)


def _run_eval(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    q_cams: torch.Tensor,
    g_cams: torch.Tensor,
    device: torch.device,
    use_rerank: bool,
) -> tuple[dict, dict]:
    """Compute mAP + CMC metrics, optionally with k-reciprocal reranking.

    Args:
        q_feats / g_feats: (N, D) L2-normalised feature tensors.
        q_pids / g_pids:   (N,) person ID tensors.
        q_cams / g_cams:   (N,) camera ID tensors.
        device:            Computation device (for reranking).
        use_rerank:        Whether to compute reranked metrics.

    Returns:
        (base_metrics, rerank_metrics) — rerank_metrics is {} if use_rerank=False.
    """
    q_feats = F.normalize(q_feats, p=2, dim=-1)
    g_feats = F.normalize(g_feats, p=2, dim=-1)
    dist = 1.0 - torch.matmul(q_feats, g_feats.T)
    base = compute_metrics(dist, q_pids, g_pids, q_cams, g_cams)

    rr: dict = {}
    if use_rerank:
        rr_dist = k_reciprocal_rerank(q_feats.to(device), g_feats.to(device)).cpu()
        rr = compute_metrics(rr_dist, q_pids, g_pids, q_cams, g_cams)
    return base, rr


def _measure_ms_baseline(model, device: torch.device, n: int = 50) -> float:
    dummy = torch.randn(1, 3, 256, 128, device=device)
    cam = torch.tensor([0], device=device)
    for _ in range(10):
        model(dummy, cam_label=cam)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(dummy, cam_label=cam)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


@torch.no_grad()
def _measure_ms_custom(
    model: CLIPReIDPedestrianModel, device: torch.device, fp16: bool, n: int = 50
) -> float:
    dummy = torch.randn(1, 3, 224, 224, device=device)
    cam = torch.tensor([1], device=device)
    view = torch.tensor([0], device=device)
    use_fp16 = fp16 and device.type == "cuda"
    for _ in range(10):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            model.extract_features(dummy, cam, view)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            model.extract_features(dummy, cam, view)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ── Report ─────────────────────────────────────────────────────────────────────


def _print_table(results: list[EvalResult]) -> None:
    header = (
        f"\n{'Model':<40} {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6} "
        f"{'mAP+RR':>8} {'R1+RR':>7} {'Params':>8} {'ms/img':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"  {r.model_name:<38} {r.mAP:>5.1f}% {r.rank1:>5.1f}% "
            f"{r.rank5:>5.1f}% {r.rank10:>5.1f}% "
            f"{r.mAP_rr:>7.1f}% {r.rank1_rr:>6.1f}% "
            f"{r.num_params_m:>7.1f}M {r.inference_ms:>7.1f}"
        )
    print(sep)

    if len(results) == 2:
        b, c = results[0], results[1]
        print(
            f"\n  Delta (Custom − Baseline):  "
            f"mAP {c.mAP - b.mAP:+.1f}%  "
            f"R1 {c.rank1 - b.rank1:+.1f}%  "
            f"mAP+RR {c.mAP_rr - b.mAP_rr:+.1f}%  "
            f"R1+RR {c.rank1_rr - b.rank1_rr:+.1f}%"
        )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline CLIP-ReID and custom ViT-CLIP-ReID-SIE-OLP on Market-1501"
    )
    parser.add_argument("--market1501-root", type=Path, required=True)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_rerank = not args.no_rerank
    fp16 = not args.no_fp16
    print(f"Device: {device}  fp16={fp16}  rerank={use_rerank}")

    results: list[EvalResult] = []

    # ── Baseline transform: CLIP normalisation, 256×128 ───────────────────────
    baseline_tfm = T.Compose([
        T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ── Baseline ──────────────────────────────────────────────────────────────
    if _BASELINE_CKPT.exists():
        print(f"\n[Baseline] Loading checkpoint: {_BASELINE_CKPT.name}")
        b_model = _load_baseline(device)
        n_params_b = sum(p.numel() for p in b_model.parameters()) / 1e6
        print(f"  Parameters: {n_params_b:.1f}M")

        q_ds = Market1501(args.market1501_root, split="query", transform=baseline_tfm, remap_pids=False)
        g_ds = Market1501(args.market1501_root, split="gallery", transform=baseline_tfm, remap_pids=False)
        print(f"  query: {len(q_ds._samples):,} images  gallery: {len(g_ds._samples):,} images")

        q_loader = DataLoader(q_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
        g_loader = DataLoader(g_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")

        print("  Extracting query features…")
        q_feats, q_pids, q_cams = _extract_baseline(b_model, q_loader, device)
        print("  Extracting gallery features…")
        g_feats, g_pids, g_cams = _extract_baseline(b_model, g_loader, device)

        if use_rerank:
            print("  Running k-reciprocal re-ranking…")
        base_m, rr_m = _run_eval(q_feats, g_feats, q_pids, g_pids, q_cams, g_cams,
                                  device, use_rerank)

        ms = _measure_ms_baseline(b_model, device)

        b_result = EvalResult(
            model_name="Baseline (CLIP-ReID)",
            mAP=base_m["mAP"], rank1=base_m["rank1"],
            rank5=base_m["rank5"], rank10=base_m["rank10"],
            mAP_rr=rr_m.get("mAP", 0.0), rank1_rr=rr_m.get("rank1", 0.0),
            num_params_m=n_params_b, inference_ms=ms,
        )
        results.append(b_result)

        # Save JSON cache for run_experiment.py --skip-baseline
        _BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE_JSON.write_text(json.dumps(asdict(b_result), indent=2), encoding="utf-8")
        print(f"  Baseline cache saved → {_BASELINE_JSON}")

        del b_model  # free GPU memory before loading custom model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"\n[Baseline] Checkpoint not found: {_BASELINE_CKPT}")

    # ── Custom model ──────────────────────────────────────────────────────────
    if _CUSTOM_CKPT.exists():
        print(f"\n[Custom] Loading checkpoint: {_CUSTOM_CKPT.name}")
        c_model, n_params_c = _load_custom(device)
        print(f"  Parameters: {n_params_c:.1f}M")

        val_tfm = build_val_transform()  # 224×224, ImageNet normalisation
        q_ds = Market1501(args.market1501_root, split="query", transform=val_tfm, remap_pids=False)
        g_ds = Market1501(args.market1501_root, split="gallery", transform=val_tfm, remap_pids=False)
        print(f"  query: {len(q_ds._samples):,} images  gallery: {len(g_ds._samples):,} images")

        q_loader = DataLoader(q_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
        g_loader = DataLoader(g_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")

        print("  Extracting query features…")
        q_feats, q_pids, q_cams = _extract_custom(c_model, q_loader, device, fp16)
        print("  Extracting gallery features…")
        g_feats, g_pids, g_cams = _extract_custom(c_model, g_loader, device, fp16)

        if use_rerank:
            print("  Running k-reciprocal re-ranking…")
        base_m, rr_m = _run_eval(q_feats, g_feats, q_pids, g_pids, q_cams, g_cams,
                                  device, use_rerank)

        ms = _measure_ms_custom(c_model, device, fp16)

        results.append(EvalResult(
            model_name="Custom (ViT-CLIP-ReID-SIE-OLP)",
            mAP=base_m["mAP"], rank1=base_m["rank1"],
            rank5=base_m["rank5"], rank10=base_m["rank10"],
            mAP_rr=rr_m.get("mAP", 0.0), rank1_rr=rr_m.get("rank1", 0.0),
            num_params_m=n_params_c, inference_ms=ms,
        ))
    else:
        print(f"\n[Custom] Checkpoint not found: {_CUSTOM_CKPT}")

    # ── Comparison table ──────────────────────────────────────────────────────
    if results:
        print("\n=== Results ===")
        _print_table(results)
    else:
        print("\n[warn] No checkpoints found — check paths above.")


if __name__ == "__main__":
    main()
