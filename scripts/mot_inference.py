"""MOT Inference — Re-ID feature extraction and person matching on MOT sequences.

Loads a trained CLIPReIDPedestrianModel checkpoint and runs inference on
crops extracted from a MOT17/MOT20 detection file.

Usage examples:
    # Match persons across two frames of a MOT sequence
    python scripts/mot_inference.py \\
        --checkpoint output/full/stage2_best.pth \\
        --mot-seq C:/pedestrian_detection/data/MOT17_human/MOT17-04-SDP \\
        --frames 1 30

    # Extract features for all detections in a sequence (save to .npy)
    python scripts/mot_inference.py \\
        --checkpoint output/full/stage2_best.pth \\
        --mot-seq C:/pedestrian_detection/data/MOT17_human/MOT17-04-SDP \\
        --extract-all --output-dir output/mot_feats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.transforms import build_val_transform
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel

# ── MOT camera map (same as research.md) ──────────────────────────────────────
MOT_CAM_MAP: dict[str, int] = {
    "MOT17-02": 10, "MOT17-04": 11, "MOT17-05": 12, "MOT17-09": 13,
    "MOT17-10": 14, "MOT17-11": 15, "MOT17-13": 16,
    "MOT20-01": 17, "MOT20-02": 18, "MOT20-03": 19, "MOT20-05": 20,
}

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint: Path, device: torch.device) -> CLIPReIDPedestrianModel:
    ckpt = torch.load(checkpoint, map_location=device)
    model = CLIPReIDPedestrianModel(
        num_pids=ckpt["num_pids"],
        num_cams=ckpt["num_cams"],
        clip_name="openai/clip-vit-base-patch16",
        template="a photo of a X X X X person",
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[MOT] loaded checkpoint: num_pids={ckpt['num_pids']}  mAP={ckpt.get('best_map', '?'):.2f}%")
    return model


def parse_det_file(det_path: Path) -> dict[int, list[tuple]]:
    """Parse MOT det.txt → {frame_id: [(x, y, w, h, conf), ...]}"""
    detections: dict[int, list] = {}
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf < 0.3 or w < 32 or h < 64:  # filter small/low-conf
                continue
            detections.setdefault(frame, []).append((x, y, w, h, conf))
    return detections


def parse_gt_file(gt_path: Path) -> dict[int, list[tuple]]:
    """Parse MOT gt.txt → {frame_id: [(track_id, x, y, w, h), ...]}"""
    gt: dict[int, list] = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            class_id = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if class_id != 1 or vis < 0.25 or w < 32 or h < 64:
                continue
            gt.setdefault(frame, []).append((track_id, x, y, w, h))
    return gt


def crop_person(img: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    """Crop a person bounding box from a PIL image, clamp to image bounds."""
    W, H = img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return img.crop((0, 0, 64, 128))   # fallback
    return img.crop((x1, y1, x2, y2))


@torch.no_grad()
def extract_features(
    model: CLIPReIDPedestrianModel,
    crops: list[Image.Image],
    cam_id: int,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract L2-normalised 512-d features for a list of PIL crops."""
    tf = build_val_transform()
    feats = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack([tf(c) for c in crops[i:i + batch_size]]).to(device)
        cam_ids = torch.full((len(batch),), cam_id, dtype=torch.long, device=device)
        view_ids = torch.zeros(len(batch), dtype=torch.long, device=device)
        feat = model.extract_features(batch, cam_ids, view_ids)   # (B, 512)
        feats.append(feat.cpu())
    return torch.cat(feats, dim=0)   # (N, 512)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between two sets of L2-normed features."""
    return torch.matmul(a, b.T)   # (Na, Nb)


# ── Main tasks ────────────────────────────────────────────────────────────────

def match_two_frames(
    model: CLIPReIDPedestrianModel,
    seq_dir: Path,
    frame_a: int,
    frame_b: int,
    cam_id: int,
    device: torch.device,
) -> None:
    """Show Re-ID similarity between all detections in frame_a vs frame_b."""
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    det_path = seq_dir / "det" / "det.txt"

    src_path = gt_path if gt_path.exists() else det_path
    boxes = parse_gt_file(src_path) if gt_path.exists() else parse_det_file(det_path)

    def load_crops(frame_id: int) -> tuple[list, list]:
        frame_file = img_dir / f"{frame_id:06d}.jpg"
        if not frame_file.exists():
            raise FileNotFoundError(f"Frame not found: {frame_file}")
        img = Image.open(frame_file).convert("RGB")
        frame_boxes = boxes.get(frame_id, [])
        if not frame_boxes:
            raise ValueError(f"No detections for frame {frame_id}")
        crops, ids = [], []
        for entry in frame_boxes:
            if len(entry) == 5:   # GT: (track_id, x, y, w, h)
                tid, x, y, w, h = entry
                crops.append(crop_person(img, x, y, w, h))
                ids.append(tid)
            else:                  # Det: (x, y, w, h, conf)
                x, y, w, h, _ = entry
                crops.append(crop_person(img, x, y, w, h))
                ids.append(-1)
        return crops, ids

    print(f"\n[MOT] Sequence : {seq_dir.name}")
    print(f"[MOT] Frame A  : {frame_a}  →  Frame B: {frame_b}")
    print(f"[MOT] Cam ID   : {cam_id}")

    crops_a, ids_a = load_crops(frame_a)
    crops_b, ids_b = load_crops(frame_b)

    feats_a = extract_features(model, crops_a, cam_id, device)
    feats_b = extract_features(model, crops_b, cam_id, device)

    sim = cosine_similarity_matrix(feats_a, feats_b)   # (Na, Nb)

    print(f"\n[MOT] Detections in frame {frame_a}: {len(crops_a)}")
    print(f"[MOT] Detections in frame {frame_b}: {len(crops_b)}")
    print(f"\n[MOT] Cosine similarity matrix (rows=frameA, cols=frameB):")
    print(f"      " + "  ".join(f"B{j:02d}" for j in range(len(crops_b))))
    for i in range(len(crops_a)):
        row = "  ".join(f"{sim[i,j]:.2f}" for j in range(len(crops_b)))
        match_j = sim[i].argmax().item()
        label = f"(→B{match_j:02d})"
        gt_label = f" GT={ids_a[i]}" if ids_a[i] != -1 else ""
        print(f"  A{i:02d}{gt_label}  {row}  {label}")

    # Best match accuracy (if GT available)
    if ids_a[0] != -1 and ids_b[0] != -1:
        correct = 0
        total = 0
        for i, tid_a in enumerate(ids_a):
            if tid_a in ids_b:
                j_gt = ids_b.index(tid_a)
                j_pred = sim[i].argmax().item()
                correct += int(j_pred == j_gt)
                total += 1
        if total > 0:
            print(f"\n[MOT] Top-1 match accuracy (GT tracklets present in both frames): "
                  f"{correct}/{total} = {100*correct/total:.1f}%")


def extract_all_features(
    model: CLIPReIDPedestrianModel,
    seq_dir: Path,
    cam_id: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Extract and save features for every detection in the sequence."""
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    det_path = seq_dir / "det" / "det.txt"

    src_path = gt_path if gt_path.exists() else det_path
    use_gt = gt_path.exists()
    boxes = parse_gt_file(src_path) if use_gt else parse_det_file(det_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_feats, all_frame_ids, all_track_ids = [], [], []
    frame_ids = sorted(boxes.keys())

    print(f"[MOT] Extracting features for {len(frame_ids)} frames in {seq_dir.name}...")
    for frame_id in frame_ids:
        frame_file = img_dir / f"{frame_id:06d}.jpg"
        if not frame_file.exists():
            continue
        img = Image.open(frame_file).convert("RGB")
        frame_boxes = boxes[frame_id]
        crops, tids = [], []
        for entry in frame_boxes:
            if use_gt:
                tid, x, y, w, h = entry
            else:
                x, y, w, h, _ = entry
                tid = -1
            crops.append(crop_person(img, x, y, w, h))
            tids.append(tid)

        if not crops:
            continue

        feats = extract_features(model, crops, cam_id, device)
        all_feats.append(feats.numpy())
        all_frame_ids.extend([frame_id] * len(crops))
        all_track_ids.extend(tids)

        if frame_id % 50 == 0:
            print(f"  frame {frame_id}/{frame_ids[-1]}  crops={len(crops)}")

    feats_np = np.concatenate(all_feats, axis=0)
    out_feats = output_dir / f"{seq_dir.name}_feats.npy"
    out_meta = output_dir / f"{seq_dir.name}_meta.npy"
    np.save(out_feats, feats_np)
    np.save(out_meta, np.array(list(zip(all_frame_ids, all_track_ids))))

    print(f"\n[MOT] Saved {feats_np.shape[0]} feature vectors → {out_feats}")
    print(f"[MOT] Saved frame/track metadata → {out_meta}")

    # Quick same-person retrieval check (if GT)
    if use_gt and -1 not in all_track_ids:
        _quick_retrieval_check(feats_np, np.array(all_track_ids))


def _quick_retrieval_check(feats: np.ndarray, track_ids: np.ndarray) -> None:
    """Compute Rank-1 accuracy: for each crop, check if top match has same track_id."""
    feats_t = torch.from_numpy(feats)
    sim = torch.matmul(feats_t, feats_t.T)
    sim.fill_diagonal_(-2.0)   # exclude self

    top1 = sim.argmax(dim=1).numpy()
    correct = (track_ids[top1] == track_ids).sum()
    total = len(track_ids)
    print(f"[MOT] Quick Rank-1 (same tracklet): {correct}/{total} = {100*correct/total:.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def visualize_retrieval(
    query_img: Image.Image,
    query_pid: int,
    query_cam: int,
    gallery_imgs: list[Image.Image],
    gallery_pids: list[int],
    gallery_sims: list[float],
    topk: int,
    save_path: Path,
    query_idx: int,
) -> None:
    """Save a grid: [Query | Top-1 | Top-2 | ... Top-K] with color borders."""
    THUMB = (96, 192)   # w, h per image
    BORDER = 6
    FONT_H = 18
    PAD = 4

    n_cols = topk + 1
    W = n_cols * (THUMB[0] + 2 * BORDER + PAD) + PAD
    H = THUMB[1] + 2 * BORDER + FONT_H + 2 * PAD

    canvas = Image.new("RGB", (W, H), (40, 40, 40))

    def paste_with_border(img: Image.Image, col: int, color: tuple) -> None:
        thumb = img.resize(THUMB, Image.BILINEAR)
        x = PAD + col * (THUMB[0] + 2 * BORDER + PAD)
        y = PAD
        # Draw border rectangle
        bordered = Image.new("RGB", (THUMB[0] + 2*BORDER, THUMB[1] + 2*BORDER), color)
        bordered.paste(thumb, (BORDER, BORDER))
        canvas.paste(bordered, (x, y))

    # Query — blue border
    paste_with_border(query_img, 0, (70, 130, 255))

    # Gallery matches — green if correct pid, red if wrong
    for k, (gimg, gpid, gsim) in enumerate(
        zip(gallery_imgs[:topk], gallery_pids[:topk], gallery_sims[:topk])
    ):
        color = (50, 220, 50) if gpid == query_pid else (220, 50, 50)
        paste_with_border(gimg, k + 1, color)

    # Add text labels
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        cell_w = THUMB[0] + 2 * BORDER + PAD

        # Query label
        draw.text((PAD + 4, PAD + THUMB[1] + 2*BORDER + 2),
                  f"QUERY\npid={query_pid} c{query_cam}", fill=(200, 200, 255))

        for k, (gpid, gsim) in enumerate(zip(gallery_pids[:topk], gallery_sims[:topk])):
            x = PAD + (k+1) * cell_w + 4
            y = PAD + THUMB[1] + 2*BORDER + 2
            label_color = (100, 255, 100) if gpid == query_pid else (255, 100, 100)
            draw.text((x, y), f"Top{k+1} sim={gsim:.2f}\npid={gpid}", fill=label_color)
    except Exception:
        pass  # text drawing is optional

    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / f"query_{query_idx:03d}_pid{query_pid}.jpg"
    canvas.save(out_file)
    return out_file


def is_market1501(path: Path) -> bool:
    """Detect if path is a Market-1501 root (has bounding_box_train/)."""
    return (path / "bounding_box_train").exists() or (path / "query").exists()


def run_market1501_inference(
    model: CLIPReIDPedestrianModel,
    root: Path,
    device: torch.device,
    num_query: int = 10,
    topk: int = 5,
    output_dir: Path = Path("output/viz"),
) -> None:
    """Run Re-ID retrieval demo on Market-1501 query → gallery, with visualization."""
    from src.datasets.market1501 import Market1501
    import random

    query_ds = Market1501(root, split="query")
    gallery_ds = Market1501(root, split="gallery")

    print(f"\n[ReID] Market-1501 inference demo")
    print(f"[ReID] query={len(query_ds)}  gallery={len(gallery_ds)}")
    print(f"[ReID] Sampling {num_query} query images...\n")

    # Pick num_query random query samples
    indices = random.sample(range(len(query_ds)), min(num_query, len(query_ds)))

    # Load and keep original gallery images for visualization
    print("[ReID] Extracting gallery features...")
    gallery_imgs, gallery_pids, gallery_cams = [], [], []
    for i in range(len(gallery_ds)):
        s = gallery_ds._samples[i]
        gallery_imgs.append(Image.open(s.img_path).convert("RGB"))
        gallery_pids.append(s.pid)
        gallery_cams.append(s.cam_id)

    gallery_feats = extract_features(model, gallery_imgs, cam_id=1, device=device, batch_size=256)
    gallery_pids_t = torch.tensor(gallery_pids)
    gallery_cams_t = torch.tensor(gallery_cams)

    correct_r1 = 0
    viz_dir = output_dir / "retrieval_viz"

    for rank, idx in enumerate(indices):
        s = query_ds._samples[idx]
        qimg = Image.open(s.img_path).convert("RGB")
        qfeat = extract_features(model, [qimg], cam_id=s.cam_id, device=device)

        # Cosine similarity — exclude same-camera same-identity (standard ReID eval)
        sim = cosine_similarity_matrix(qfeat, gallery_feats)[0]   # (G,)
        excl = (gallery_pids_t == s.pid) & (gallery_cams_t == s.cam_id)
        sim[excl] = -2.0

        top_idx = sim.topk(topk).indices.tolist()
        top_pids  = [gallery_pids[i] for i in top_idx]
        top_sims  = [sim[i].item()   for i in top_idx]
        top_imgs  = [gallery_imgs[i] for i in top_idx]

        hit_r1 = top_pids[0] == s.pid
        correct_r1 += int(hit_r1)
        status = "✓" if hit_r1 else "✗"

        print(f"  Query {rank+1:2d}: pid={s.pid:4d} cam={s.cam_id}  "
              f"Top-{topk} pids: {top_pids}  "
              f"sims: {[f'{x:.2f}' for x in top_sims]}  {status}")

        # Save visualization grid
        out_file = visualize_retrieval(
            query_img=qimg,
            query_pid=s.pid,
            query_cam=s.cam_id,
            gallery_imgs=top_imgs,
            gallery_pids=top_pids,
            gallery_sims=top_sims,
            topk=topk,
            save_path=viz_dir,
            query_idx=rank + 1,
        )
        print(f"           → saved: {out_file}")

    print(f"\n[ReID] Rank-1 accuracy on {num_query} queries: "
          f"{correct_r1}/{num_query} = {100*correct_r1/num_query:.1f}%")
    print(f"[ReID] Visualizations saved to: {viz_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-ID inference: Market-1501 or MOT sequence")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to stage2_best.pth")
    p.add_argument("--mot-seq", type=Path, required=True,
                   help="Path to Market-1501 root OR a MOT sequence dir (e.g. MOT17-04-SDP)")
    p.add_argument("--frames", type=int, nargs=2, default=[1, 30],
                   metavar=("FRAME_A", "FRAME_B"),
                   help="[MOT only] Two frame IDs to compare (default: 1 30)")
    p.add_argument("--extract-all", action="store_true",
                   help="[MOT only] Extract features for all frames and save to .npy")
    p.add_argument("--num-query", type=int, default=10,
                   help="[Market-1501] Number of query images to test (default: 10)")
    p.add_argument("--topk", type=int, default=5,
                   help="[Market-1501] Top-K matches to show (default: 5)")
    p.add_argument("--output-dir", type=Path, default=Path("output/mot_feats"))
    p.add_argument("--cam-id", type=int, default=None,
                   help="[MOT] Camera ID override (auto-detected if not set)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ReID] device={device}")

    model = load_model(args.checkpoint, device)

    # ── Auto-detect dataset type ───────────────────────────────────────────────
    if is_market1501(args.mot_seq):
        print(f"[ReID] Detected Market-1501 at {args.mot_seq}")
        run_market1501_inference(
            model, args.mot_seq, device,
            num_query=args.num_query,
            topk=args.topk,
            output_dir=args.output_dir,
        )
    else:
        # MOT sequence mode
        cam_id = args.cam_id
        if cam_id is None:
            seq_name = args.mot_seq.name
            base = "-".join(seq_name.split("-")[:2])
            cam_id = MOT_CAM_MAP.get(base, 1)
            print(f"[ReID] auto cam_id={cam_id} for sequence '{seq_name}'")

        if args.extract_all:
            extract_all_features(model, args.mot_seq, cam_id, device, args.output_dir)
        else:
            match_two_frames(model, args.mot_seq, args.frames[0], args.frames[1], cam_id, device)
