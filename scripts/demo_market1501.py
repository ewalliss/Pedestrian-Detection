"""Market-1501 Re-ID Visual Demo.

For each query image, retrieves top-K gallery matches and draws an annotated
grid showing:
  - Cosine similarity score (the matching signal)
  - Green border  = correct match (same PID)
  - Red border    = wrong match  (different PID)
  - FPS / latency at top of each column

Similarity score explained
--------------------------
The model produces a 1024-d feature vector:
    feat = concat(BN(global_512), BN(fused_512))
then L2-normalised before matching.

The matching score is cosine similarity:
    score = feat_query · feat_gallery   ∈ [-1, 1]

Recommended thresholds (from CosineMatcher default + empirical runs5 results):
  ≥ 0.80  — high confidence same person
  0.60–0.79 — medium confidence (accept with caution)
  < 0.60  — likely different person

Usage
-----
python scripts/demo_market1501.py \\
    --checkpoint model/stage2_best_c_loss.pth \\
    --market1501-root /path/to/Market-1501-v15.09.15

Flags
-----
--num-queries N        How many query images to visualise (default 10)
--top-k K              Gallery top-K results per query (default 10)
--threshold T          Draw score in yellow if below threshold (default 0.75)
--no-rerank            Skip k-reciprocal re-ranking distance
--no-fp16              Disable FP16 (for CPU)
--output-dir PATH      Save PNG grids here instead of displaying them
--batch-size N         Feature extraction batch size (default 128)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.datasets.market1501 import Market1501
from src.datasets.transforms import build_val_transform
from src.eval.evaluate import power_normalize, alpha_query_expansion
from src.eval.reranking import k_reciprocal_rerank
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


# ── Visual constants ──────────────────────────────────────────────────────────

_GREEN = (0, 200, 0)
_RED   = (220, 30, 30)
_YELLOW = (255, 200, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREY_BG = (40, 40, 40)

_CELL_W = 80   # px per gallery cell (resized from 128×256)
_CELL_H = 120
_QUERY_W = 100
_QUERY_H = 150
_BORDER = 3
_FONT_SIZE = 11


def _load_font(size: int = _FONT_SIZE) -> ImageFont.ImageFont:
    for name in [
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device: torch.device) -> CLIPReIDPedestrianModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_pids = ckpt.get("num_pids", 751)
    num_cams = ckpt.get("num_cams", 30)
    print(f"[model] num_pids={num_pids}  num_cams={num_cams}  epoch={ckpt.get('epoch','?')}")
    model = CLIPReIDPedestrianModel(num_pids=num_pids, num_cams=num_cams)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def _extract_all(
    model: CLIPReIDPedestrianModel,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> tuple[torch.Tensor, list[int], list[int], float]:
    """Returns (feats_cpu, pids, cam_ids, ms_per_image)."""
    feats, pids_out, cams_out = [], [], []
    n_imgs = 0
    model.eval()

    t0 = time.perf_counter()
    for imgs, pids, cam_ids, view_ids in tqdm(loader, desc="  extracting", leave=False):
        imgs     = imgs.to(device)
        cam_ids  = cam_ids.to(device)
        view_ids = view_ids.to(device)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=fp16 and device.type == "cuda"
        ):
            feat = model.extract_features(imgs, cam_ids, view_ids)
        feats.append(feat.cpu().float())
        pids_out.extend(pids.tolist())
        cams_out.extend(cam_ids.cpu().tolist())
        n_imgs += imgs.shape[0]

    elapsed = time.perf_counter() - t0
    ms_per_img = elapsed / max(n_imgs, 1) * 1000
    return torch.cat(feats, dim=0), pids_out, cams_out, ms_per_img


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_cell(
    img_path: Path,
    score: float,
    correct: bool,
    cell_w: int,
    cell_h: int,
    font: ImageFont.ImageFont,
    threshold: float,
) -> Image.Image:
    """Render one gallery cell: resized image + coloured border + score label."""
    cell = Image.new("RGB", (cell_w + 2 * _BORDER, cell_h + 2 * _BORDER + 16), _BLACK)
    try:
        img = Image.open(img_path).convert("RGB").resize(
            (cell_w, cell_h), Image.BICUBIC
        )
    except Exception:
        img = Image.new("RGB", (cell_w, cell_h), (80, 80, 80))

    border_color = _GREEN if correct else _RED
    cell.paste(img, (_BORDER, _BORDER))

    draw = ImageDraw.Draw(cell)
    # border rectangle
    draw.rectangle(
        [0, 0, cell.width - 1, cell_h + 2 * _BORDER - 1],
        outline=border_color,
        width=_BORDER,
    )
    # score label
    score_color = _YELLOW if score < threshold else _WHITE
    draw.rectangle(
        [0, cell_h + 2 * _BORDER, cell.width - 1, cell.height - 1],
        fill=border_color,
    )
    draw.text(
        (cell.width // 2, cell_h + 2 * _BORDER + 2),
        f"{score:.3f}",
        fill=score_color,
        font=font,
        anchor="mt",
    )
    return cell


def _draw_query_cell(
    img_path: Path,
    pid: int,
    cam_id: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    header_h = 30
    cell = Image.new("RGB", (_QUERY_W + 2 * _BORDER, _QUERY_H + 2 * _BORDER + header_h), _GREY_BG)
    try:
        img = Image.open(img_path).convert("RGB").resize(
            (_QUERY_W, _QUERY_H), Image.BICUBIC
        )
    except Exception:
        img = Image.new("RGB", (_QUERY_W, _QUERY_H), (80, 80, 80))
    cell.paste(img, (_BORDER, header_h))
    draw = ImageDraw.Draw(cell)
    draw.rectangle(
        [0, header_h, cell.width - 1, cell.height - 1],
        outline=(255, 165, 0),
        width=_BORDER,
    )
    draw.text(
        (cell.width // 2, header_h // 2),
        f"QUERY",
        fill=(255, 165, 0),
        font=font,
        anchor="mm",
    )
    draw.text(
        (cell.width // 2, _QUERY_H + header_h + _BORDER + 2),
        f"PID {pid} cam{cam_id}",
        fill=_WHITE,
        font=font,
        anchor="mt",
    )
    return cell


def _build_row(
    query_path: Path,
    query_pid: int,
    query_cam: int,
    top_k_paths: list[Path],
    top_k_pids: list[int],
    top_k_scores: list[float],
    threshold: float,
    font: ImageFont.ImageFont,
) -> Image.Image:
    q_cell = _draw_query_cell(query_path, query_pid, query_cam, font)
    g_cells = [
        _draw_cell(p, s, pid == query_pid, _CELL_W, _CELL_H, font, threshold)
        for p, pid, s in zip(top_k_paths, top_k_pids, top_k_scores)
    ]

    row_h = max(c.height for c in [q_cell] + g_cells)
    row_w = q_cell.width + 8 + sum(c.width + 2 for c in g_cells)
    row = Image.new("RGB", (row_w, row_h), _GREY_BG)
    row.paste(q_cell, (0, (row_h - q_cell.height) // 2))
    x = q_cell.width + 8
    for c in g_cells:
        row.paste(c, (x, (row_h - c.height) // 2))
        x += c.width + 2
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    fp16 = not args.no_fp16 and device.type == "cuda"
    print(f"[demo] device={device}  fp16={fp16}  checkpoint={ckpt_path.name}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _load_model(ckpt_path, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    tfm = build_val_transform()
    market_root = Path(args.market1501_root)
    q_ds = Market1501(market_root, split="query",   transform=tfm, remap_pids=False)
    g_ds = Market1501(market_root, split="gallery", transform=tfm, remap_pids=False)
    print(f"[data]  query: {len(q_ds):,}  gallery: {len(g_ds):,}")

    def _make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    print("[extract] query features…")
    q_feats, q_pids, q_cams, q_ms = _extract_all(model, _make_loader(q_ds), device, fp16)
    print("[extract] gallery features…")
    g_feats, g_pids, g_cams, g_ms = _extract_all(model, _make_loader(g_ds), device, fp16)

    q_fps = 1000.0 / max(q_ms, 1e-6)
    g_fps = 1000.0 / max(g_ms, 1e-6)
    print(f"[latency] query={q_ms:.2f} ms/img ({q_fps:.1f} FPS)  gallery={g_ms:.2f} ms/img ({g_fps:.1f} FPS)")

    # ── Post-processing: power norm → L2 → optional QE ────────────────────────
    q_feats = F.normalize(power_normalize(q_feats, 0.5), p=2, dim=-1)
    g_feats = F.normalize(power_normalize(g_feats, 0.5), p=2, dim=-1)
    if not args.no_qe:
        q_feats = alpha_query_expansion(q_feats, g_feats, k=5, alpha=3.0)

    # ── Distance matrix ────────────────────────────────────────────────────────
    if args.no_rerank:
        dist_mat = 1.0 - torch.matmul(q_feats, g_feats.T)  # (Q, G)
        print("[dist] cosine distance (no rerank)")
    else:
        dist_mat = k_reciprocal_rerank(q_feats, g_feats, k1=20, k2=6, lambda_=0.7)
        print("[dist] k-reciprocal rerank distance")

    # ── Matching score for display = cosine similarity (intuitive) ────────────
    # Even if we used rerank-distance for ranking, show cosine sim as the score.
    # Score ∈ [-1, 1]; recommended threshold for acceptance: ≥ 0.75.
    # Higher = more similar. 1.0 = identical features.
    cosine_sim = torch.matmul(q_feats, g_feats.T)  # (Q, G), always show cosine

    # ── Select queries ─────────────────────────────────────────────────────────
    import random
    random.seed(42)
    # Prefer queries that have at least one correct gallery hit
    valid_q = [
        i for i in range(len(q_pids))
        if any(
            g_pids[j] == q_pids[i] and g_cams[j] != q_cams[i]
            for j in range(len(g_pids))
        )
    ]
    if not valid_q:
        valid_q = list(range(len(q_pids)))
    chosen = random.sample(valid_q, min(args.num_queries, len(valid_q)))
    print(f"[demo] visualising {len(chosen)} queries  top-K={args.top_k}")
    print(f"\n  Matching score: cosine similarity ∈ [-1, 1]")
    print(f"  Recommended threshold: ≥ {args.threshold:.2f}")
    print(f"  Green = correct match  |  Red = wrong match  |  Yellow score = below threshold\n")

    # ── Build gallery sample list (for path lookup) ────────────────────────────
    g_samples = g_ds._samples
    q_samples = q_ds._samples

    font = _load_font(_FONT_SIZE)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, q_idx in enumerate(chosen):
        q_pid  = q_pids[q_idx]
        q_cam  = q_cams[q_idx]
        q_path = q_samples[q_idx].img_path

        # Sort by distance (ascending), exclude same-camera same-pid distractors
        dist_row = dist_mat[q_idx].clone()
        for j in range(len(g_pids)):
            if g_pids[j] == q_pid and g_cams[j] == q_cam:
                dist_row[j] = 1e9   # mask out same-cam same-id

        sorted_idx = dist_row.argsort()[:args.top_k]
        top_paths  = [g_samples[j].img_path for j in sorted_idx]
        top_pids   = [g_pids[j] for j in sorted_idx]
        top_scores = [float(cosine_sim[q_idx, j]) for j in sorted_idx]

        n_correct = sum(1 for p in top_pids if p == q_pid)
        print(
            f"  [{idx+1:2d}/{len(chosen)}]  PID={q_pid:4d}  cam={q_cam}"
            f"  top-{args.top_k}: {n_correct}/{args.top_k} correct"
            f"  best_score={top_scores[0]:.3f}"
        )

        row = _build_row(
            q_path, q_pid, q_cam,
            top_paths, top_pids, top_scores,
            args.threshold, font,
        )
        rows.append(row)

        if output_dir:
            out_path = output_dir / f"query_{idx+1:02d}_pid{q_pid}.png"
            row.save(out_path)
            print(f"    saved → {out_path}")

    # ── Assemble grid ──────────────────────────────────────────────────────────
    total_h = sum(r.height + 4 for r in rows) + 60
    max_w   = max(r.width for r in rows)

    grid = Image.new("RGB", (max_w, total_h), _GREY_BG)
    draw = ImageDraw.Draw(grid)

    # Header info
    header_font = _load_font(13)
    draw.text(
        (8, 8),
        (
            f"V-Track  |  checkpoint: {ckpt_path.name}"
            f"  |  device: {device}"
            f"  |  latency: {q_ms:.1f} ms/img  ({q_fps:.0f} FPS)"
            f"  |  threshold: ≥{args.threshold:.2f}"
        ),
        fill=_WHITE,
        font=header_font,
    )
    draw.text(
        (8, 28),
        "Green = correct match   Red = wrong match   Yellow score = below threshold",
        fill=(180, 180, 180),
        font=font,
    )

    y = 55
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + 4

    if output_dir:
        grid_path = output_dir / "grid_all.png"
        grid.save(grid_path)
        print(f"\n[saved] {grid_path}")
    else:
        grid.show(title="V-Track Re-ID Demo — Market-1501")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Latency (feature extraction only)")
    print(f"    query    : {q_ms:.2f} ms/img  → {q_fps:.1f} FPS")
    print(f"    gallery  : {g_ms:.2f} ms/img  → {g_fps:.1f} FPS")
    print(f"  Matching score : cosine similarity ∈ [-1, 1]")
    print(f"  Threshold      : ≥ {args.threshold:.2f}  (configurable via --threshold)")
    print(f"    ≥ 0.80  — high confidence same person")
    print(f"    0.60–0.79 — medium confidence")
    print(f"    < 0.60  — likely different person")
    print(f"{'='*60}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Market-1501 Re-ID visual demo")
    p.add_argument("--checkpoint", "--ckpt", type=str,
                   default="model/stage2_best_c_loss.pth",
                   help="Path to Stage 2 checkpoint")
    p.add_argument("--market1501-root", type=str,
                   default=str(Path.home() / "Downloads" / "Market-1501-v15.09.15"),
                   help="Path to Market-1501 root directory")
    p.add_argument("--num-queries", type=int, default=10,
                   help="Number of query images to visualise")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top gallery results per query")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Cosine similarity threshold; scores below shown in yellow")
    p.add_argument("--no-rerank", action="store_true",
                   help="Skip k-reciprocal re-ranking (faster)")
    p.add_argument("--no-qe", action="store_true",
                   help="Skip alpha-query expansion")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable FP16 (for CPU)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Save PNG grids here instead of displaying")
    return p.parse_args()


if __name__ == "__main__":
    main()
