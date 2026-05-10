"""Experiment: Baseline (original CLIP-ReID) vs Custom (ViT-CLIP-ReID-SIE-OLP).

Evaluates both models on Market-1501, compares mAP / CMC@{1,5,10} with and
without k-reciprocal re-ranking, measures per-image latency, and writes a
markdown report.

Outputs
-------
model/model_baseline/baseline_metrics.json      (cached; reload with --skip-baseline)
docs/YYYY-MM-DD-experiment-results.md           (new dated file every run)

Usage
-----
python scripts/run_experiment.py \\
    --market1501-root /path/to/Market-1501-v15.09.15

Flags
-----
--skip-baseline   Load saved baseline JSON (skip re-running the model).
                  Prints an error and exits if the JSON cache is not found.
                  Without this flag, the baseline is always re-run and the
                  JSON cache is refreshed.
--no-rerank       Skip k-reciprocal re-ranking (faster).
--no-fp16         Disable FP16 for custom model (CPU runs).
--batch-size N    DataLoader batch size (default 128).
--num-workers N   DataLoader workers (default 2).
"""

from __future__ import annotations

import argparse
import datetime
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_ORIG_REID = _REPO / "original_CLIP-REID" / "CLIP-ReID"
_BASELINE_CKPT = _REPO / "model" / "model_baseline" / "Market1501_clipreid_12x12sie_ViT-B-16_60.pth"
_CUSTOM_CKPT_DEFAULT = _REPO / "model" / "custom" / "stage2_best.pth"
_BASELINE_JSON = _REPO / "model" / "model_baseline" / "baseline_metrics.json"
_DOCS_DIR = _REPO / "docs"

sys.path.insert(0, str(_REPO))

from src.datasets.market1501 import Market1501
from src.datasets.transforms import build_val_transform
from src.eval.evaluate import compute_metrics
from src.eval.reranking import k_reciprocal_rerank
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel

# ── Data containers ────────────────────────────────────────────────────────────

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


# ── Dataset validation ─────────────────────────────────────────────────────────

def _validate_dataset(root: Path) -> None:
    """Check that the Market-1501 directory structure exists and is non-empty.

    Args:
        root: Path to Market-1501-v15.09.15 root directory.

    Raises:
        FileNotFoundError: If root or any required subdirectory is missing / empty.
    """
    print(f"\n[Dataset] Root: {root}")

    # (subdir, min expected file count for a rough sanity check)
    required: list[tuple[str, int]] = [
        ("query", 3000),
        ("bounding_box_test", 15000),
        ("bounding_box_train", 10000),
    ]

    if not root.is_dir():
        raise FileNotFoundError(
            f"Market-1501 root not found: {root}\n"
            "Pass the correct path with --market1501-root"
        )

    all_ok = True
    for subdir, min_count in required:
        d = root / subdir
        if not d.is_dir():
            print(f"  [MISSING] {subdir}/")
            all_ok = False
            continue
        n = len(list(d.glob("*.jpg")))
        status = "[OK]" if n >= min_count else "[WARN]"
        print(f"  {status} {subdir}/ — {n:,} jpg files (expected ≥{min_count:,})")
        if n == 0:
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(
            f"Market-1501 at {root} is incomplete.\n"
            "Re-download or fix the path with --market1501-root"
        )


def _print_split_stats(dataset: Market1501, name: str) -> None:
    """Print image / ID / camera counts for a dataset split.

    Args:
        dataset: Loaded Market1501 split.
        name:    Human-readable label (e.g. "query", "gallery").
    """
    pids = {s.pid for s in dataset._samples}
    cams = {s.cam_id for s in dataset._samples}
    print(f"  {name:8s}: {len(dataset._samples):6,} images  {len(pids):4} IDs  {len(cams)} cameras")


# ── Model adapters ─────────────────────────────────────────────────────────────

class BaselineAdapter:
    """Wraps original CLIP-ReID build_transformer for a unified feature interface.

    The original model uses:
    - Input: 256×128, CLIP normalization [0.5, 0.5, 0.5]
    - Camera IDs: 0-indexed  (Market-1501 stores 1-indexed cam IDs)
    - Output (eval): torch.cat([img_feature, img_feature_proj], dim=1) → 1280-d
    """

    def __init__(self, model, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract_features(
        self,
        imgs: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,  # noqa: ARG002 — baseline does not use view info
    ) -> torch.Tensor:
        imgs = imgs.to(self.device)
        # Market-1501 cam_ids are 1-indexed (1–6); original model expects 0-indexed
        cam_label = (cam_ids - 1).to(self.device)
        feat = self.model(imgs, cam_label=cam_label)  # (N, 1280) in eval mode
        return F.normalize(feat.float(), p=2, dim=-1)


class CustomAdapter:
    """Wraps CLIPReIDPedestrianModel for the same feature interface."""

    def __init__(self, model, device: torch.device, fp16: bool = True) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"

    @torch.no_grad()
    def extract_features(
        self,
        imgs: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> torch.Tensor:
        imgs = imgs.to(self.device)
        cam_ids = cam_ids.to(self.device)
        view_ids = view_ids.to(self.device)
        with torch.amp.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.fp16
        ):
            return self.model.extract_features(imgs, cam_ids, view_ids)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def _extract_all(
    adapter,
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, pids_out, cams_out = [], [], []
    for imgs, pids, cam_ids, view_ids in tqdm(loader, desc="  extracting", leave=False):
        f = adapter.extract_features(imgs, cam_ids, view_ids)
        feats.append(f.cpu().float())
        pids_out.append(pids)
        cams_out.append(cam_ids.cpu())
    feats_t = torch.cat(feats)
    norms = feats_t.norm(dim=-1)
    print(
        f"    features: {list(feats_t.shape)}"
        f"  norm=[{norms.min():.3f}, {norms.max():.3f}]"
        f"  NaN={torch.isnan(feats_t).any().item()}"
    )
    return feats_t, torch.cat(pids_out), torch.cat(cams_out)


def _run_eval(
    adapter,
    q_loader: DataLoader,
    g_loader: DataLoader,
    device: torch.device,
    use_rerank: bool,
) -> tuple[dict, dict]:
    print("  Extracting query features…")
    q_feats, q_pids, q_cams = _extract_all(adapter, q_loader)
    print("  Extracting gallery features…")
    g_feats, g_pids, g_cams = _extract_all(adapter, g_loader)

    q_feats = F.normalize(q_feats, p=2, dim=-1)
    g_feats = F.normalize(g_feats, p=2, dim=-1)
    dist = 1.0 - torch.matmul(q_feats, g_feats.T)

    # PID overlap and distance sanity
    q_pid_set = set(q_pids.tolist())
    g_pid_set = set(g_pids.tolist())
    overlap = len(q_pid_set & g_pid_set)
    print(
        f"  PIDs — query: {len(q_pid_set)}  gallery: {len(g_pid_set)}  overlap: {overlap}"
    )
    print(
        f"  Distance matrix: {list(dist.shape)}"
        f"  range=[{dist.min():.3f}, {dist.max():.3f}]"
        f"  mean={dist.mean():.3f}"
    )

    base = compute_metrics(dist, q_pids, g_pids, q_cams, g_cams)
    rr: dict = {}
    if use_rerank:
        rr_device = torch.device("cpu") if device.type == "mps" else device
        rr_dist = k_reciprocal_rerank(q_feats.to(rr_device), g_feats.to(rr_device)).cpu()
        rr = compute_metrics(rr_dist, q_pids, g_pids, q_cams, g_cams)

    return base, rr


def _measure_ms(adapter, device: torch.device, input_size: tuple, n: int = 50) -> float:
    dummy = torch.randn(1, *input_size, device=device)
    cam = torch.tensor([1], device=device)
    view = torch.tensor([0], device=device)
    for _ in range(10):  # warmup
        adapter.extract_features(dummy, cam, view)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        adapter.extract_features(dummy, cam, view)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms/image


# ── Report writer ──────────────────────────────────────────────────────────────

def _write_report(
    results: list[EvalResult],
    market1501_root: Path,
    device_str: str,
) -> Path:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def _d(a: float, b: float) -> str:
        delta = a - b
        return f"{'+'if delta >= 0 else ''}{delta:.1f}%"

    rows = "\n".join(
        f"| **{r.model_name}** "
        f"| {r.mAP:.1f}% "
        f"| {r.rank1:.1f}% "
        f"| {r.rank5:.1f}% "
        f"| {r.rank10:.1f}% "
        f"| {r.mAP_rr:.1f}% "
        f"| {r.rank1_rr:.1f}% "
        f"| {r.num_params_m:.1f} "
        f"| {r.inference_ms:.1f} |"
        for r in results
    )

    delta_section = ""
    if len(results) == 2:
        b, c = results[0], results[1]
        delta_section = (
            "\n---\n"
            "\n## Delta: Custom − Baseline\n"
            "\n| mAP | Rank-1 | mAP+RR | Rank-1+RR |\n"
            "|-----|--------|--------|----------|\n"
            f"| {_d(c.mAP, b.mAP)} "
            f"| {_d(c.rank1, b.rank1)} "
            f"| {_d(c.mAP_rr, b.mAP_rr)} "
            f"| {_d(c.rank1_rr, b.rank1_rr)} |"
        )

    report = f"""\
# Experiment Results: Baseline vs Custom Model

**Date**: {now}
**Dataset**: Market-1501
**Device**: `{device_str}`
**Market-1501 root**: `{market1501_root}`

---

## Metrics (Market-1501)

| Model | mAP | Rank-1 | Rank-5 | Rank-10 | mAP+RR | Rank-1+RR | Params (M) | Speed (ms/img) |
|-------|-----|--------|--------|---------|--------|-----------|------------|----------------|
{rows}
{delta_section}

---

## Checkpoint Details

| | Checkpoint | Architecture | Input size |
|-|-----------|--------------|------------|
| **Baseline** | `model/model_baseline/Market1501_clipreid_12x12sie_ViT-B-16_60.pth` | Original CLIP-ReID ViT-B/16, stride 12×12, SIE(cam), feature 1280-d | 256×128 |
| **Custom** | `model/custom/stage2_best.pth` | ViT-CLIP-ReID-SIE-OLP (HuggingFace ViT-B/16-patch16), OLP top-k, feature 512-d | 224×224 |

---

*Generated by `scripts/run_experiment.py`*
"""

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    report_path = _DOCS_DIR / f"{date_str}-experiment-results.md"
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport → {report_path}")
    return report_path


# ── Baseline loading ───────────────────────────────────────────────────────────

def _load_baseline(device: torch.device) -> BaselineAdapter:
    """Build and load the original CLIP-ReID model from the baseline checkpoint."""
    sys.path.insert(0, str(_ORIG_REID))
    try:
        from config import cfg as orig_cfg  # yacs CfgNode from original repo
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

    # Market-1501: 751 train IDs, 6 cameras, no view annotations
    model = orig_make_model(orig_cfg, num_class=751, camera_num=6, view_num=1)
    model.load_param(str(_BASELINE_CKPT))
    return BaselineAdapter(model, device)


def _load_custom(ckpt_path: Path, device: torch.device, fp16: bool) -> tuple[CustomAdapter, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_pids = ckpt.get("num_pids", 751)
    num_cams = ckpt.get("num_cams", 30)
    print(f"  Checkpoint metadata: num_pids={num_pids}  num_cams={num_cams}")
    model = CLIPReIDPedestrianModel(num_pids=num_pids, num_cams=num_cams)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    return CustomAdapter(model, device, fp16=fp16), _count_params_m(model)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline vs Custom model experiment on Market-1501"
    )
    parser.add_argument(
        "--market1501-root",
        type=Path,
        required=True,
        help="Path to Market-1501-v15.09.15 root directory (required)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Load saved baseline JSON cache instead of running the model. "
            "Exits with an error if the cache does not exist. "
            "Without this flag the baseline is always re-run and the cache refreshed."
        ),
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--custom-ckpt",
        type=Path,
        default=None,
        help="Path to custom checkpoint (default: model/custom/stage2_best.pth)",
    )
    args = parser.parse_args()

    custom_ckpt = args.custom_ckpt or _CUSTOM_CKPT_DEFAULT

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_rerank = not args.no_rerank
    fp16 = not args.no_fp16
    print(f"Device: {device}  fp16={fp16}  rerank={use_rerank}")

    _validate_dataset(args.market1501_root)

    results: list[EvalResult] = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.skip_baseline:
        if _BASELINE_JSON.exists():
            print(f"\n[Baseline] Loading cached metrics from {_BASELINE_JSON.name}")
            results.append(EvalResult(**json.loads(_BASELINE_JSON.read_text())))
            r = results[-1]
            print(
                f"  mAP={r.mAP:.1f}%  R1={r.rank1:.1f}%  "
                f"[+RR] mAP={r.mAP_rr:.1f}%  R1={r.rank1_rr:.1f}%"
            )
        else:
            print(f"\n[Baseline] Skipped (no cached metrics at {_BASELINE_JSON})")
    elif _BASELINE_CKPT.exists():
        print(f"\n[Baseline] Loading checkpoint: {_BASELINE_CKPT.name}")
        adapter = _load_baseline(device)

        # Baseline was trained with CLIP normalisation [0.5, 0.5, 0.5], 256×128
        baseline_tfm = T.Compose(
            [
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        q_ds = Market1501(args.market1501_root, split="query", transform=baseline_tfm, remap_pids=False)
        g_ds = Market1501(args.market1501_root, split="gallery", transform=baseline_tfm, remap_pids=False)
        _print_split_stats(q_ds, "query")
        _print_split_stats(g_ds, "gallery")

        q_loader = DataLoader(
            q_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        g_loader = DataLoader(
            g_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        print("  Evaluating baseline…")
        base_m, rr_m = _run_eval(adapter, q_loader, g_loader, device, use_rerank)
        ms = _measure_ms(adapter, device, input_size=(3, 256, 128))
        n_params = _count_params_m(adapter.model)

        result = EvalResult(
            model_name="Baseline (CLIP-ReID)",
            mAP=base_m["mAP"],
            rank1=base_m["rank1"],
            rank5=base_m["rank5"],
            rank10=base_m["rank10"],
            mAP_rr=rr_m.get("mAP", 0.0),
            rank1_rr=rr_m.get("rank1", 0.0),
            num_params_m=n_params,
            inference_ms=ms,
        )
        results.append(result)

        _BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE_JSON.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print(f"  Baseline saved → {_BASELINE_JSON}")
        print(
            f"  mAP={result.mAP:.1f}%  R1={result.rank1:.1f}%  "
            f"[+RR] mAP={result.mAP_rr:.1f}%  R1={result.rank1_rr:.1f}%"
        )
    else:
        print(f"\n[Baseline] Checkpoint not found: {_BASELINE_CKPT}")

    # ── Custom model ──────────────────────────────────────────────────────────
    if custom_ckpt.exists():
        print(f"\n[Custom] Loading checkpoint: {custom_ckpt.name}")
        adapter, n_params = _load_custom(custom_ckpt, device, fp16=fp16)

        val_tfm = build_val_transform()  # 224×224, ImageNet norm
        q_ds = Market1501(args.market1501_root, split="query", transform=val_tfm, remap_pids=False)
        g_ds = Market1501(args.market1501_root, split="gallery", transform=val_tfm, remap_pids=False)
        _print_split_stats(q_ds, "query")
        _print_split_stats(g_ds, "gallery")

        q_loader = DataLoader(
            q_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        g_loader = DataLoader(
            g_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        print("  Evaluating custom model…")
        base_m, rr_m = _run_eval(adapter, q_loader, g_loader, device, use_rerank)
        ms = _measure_ms(adapter, device, input_size=(3, 224, 224))

        ckpt_label = custom_ckpt.stem
        result = EvalResult(
            model_name=f"Custom ({ckpt_label})",
            mAP=base_m["mAP"],
            rank1=base_m["rank1"],
            rank5=base_m["rank5"],
            rank10=base_m["rank10"],
            mAP_rr=rr_m.get("mAP", 0.0),
            rank1_rr=rr_m.get("rank1", 0.0),
            num_params_m=n_params,
            inference_ms=ms,
        )
        results.append(result)
        print(
            f"  mAP={result.mAP:.1f}%  R1={result.rank1:.1f}%  "
            f"[+RR] mAP={result.mAP_rr:.1f}%  R1={result.rank1_rr:.1f}%"
        )
    else:
        print(f"\n[Custom] Checkpoint not found: {custom_ckpt}")

    # ── Report ────────────────────────────────────────────────────────────────
    if not results:
        print("\n[warn] No results — check checkpoint paths and re-run.")
        return

    report_path = _write_report(results, args.market1501_root, str(device))

    print(f"\n=== Summary ({report_path.name}) ===")
    for r in results:
        print(f"  {r.model_name:45s}  mAP={r.mAP:.1f}%  R1={r.rank1:.1f}%")


if __name__ == "__main__":
    main()
