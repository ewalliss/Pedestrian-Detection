"""Video inference — YOLOv8 person detection + CLIP-ReID re-identification.

Reads a video file, detects persons per frame with YOLOv8, extracts Re-ID
features with a trained CLIPReIDPedestrianModel, and writes an annotated
output video with persistent identity labels.

Usage:
    python scripts/video_inference.py \
        --checkpoint runs/stage2_best.pth \
        --video input.mp4 \
        --output output_reid.mp4

    # With custom threshold and camera ID
    python scripts/video_inference.py \
        --checkpoint runs/stage2_best.pth \
        --video input.mp4 \
        --output output_reid.mp4 \
        --threshold 0.5 \
        --cam-id 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.transforms import build_val_transform
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel


# ── Color palette (32 distinct hues via HSV) ──────────────────────────────────

def _build_palette(n: int = 32) -> list[tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hsv = np.uint8([[[int(i * 180 / n), 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


_PALETTE = _build_palette()


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(checkpoint: Path, device: torch.device) -> CLIPReIDPedestrianModel:
    """Load a Stage 2 checkpoint into eval mode."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CLIPReIDPedestrianModel(
        num_pids=ckpt["num_pids"],
        num_cams=ckpt["num_cams"],
        clip_name="openai/clip-vit-base-patch16",
        template="a photo of a X X X X person",
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[ReID] loaded checkpoint  pids={ckpt['num_pids']}  cams={ckpt['num_cams']}  "
          f"mAP={ckpt.get('best_map', '?')}")
    return model


@torch.no_grad()
def extract_features(
    model: CLIPReIDPedestrianModel,
    crops: list[Image.Image],
    cam_id: int,
    device: torch.device,
    transform,
) -> torch.Tensor:
    """Extract L2-normalised 512-d features for a list of PIL crops."""
    if not crops:
        return torch.empty(0, 512)
    batch = torch.stack([transform(c) for c in crops]).to(device)
    cam_ids = torch.full((len(batch),), cam_id, dtype=torch.long, device=device)
    view_ids = torch.zeros(len(batch), dtype=torch.long, device=device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        feats = model.extract_features(batch, cam_ids, view_ids)
    return F.normalize(feats.float(), p=2, dim=-1)


# ── Simple tracker (cosine gallery matching + temporal smoothing) ─────────────

class SimpleReIDTracker:
    """Frame-by-frame Re-ID tracker using cosine similarity against a gallery."""

    def __init__(self, threshold: float = 0.5, window: int = 30) -> None:
        self._threshold = threshold
        self._window = window
        self._gallery: dict[int, list[torch.Tensor]] = {}  # gid → deque of feats
        self._next_id = 1

    def update(self, feats: torch.Tensor) -> list[int]:
        """Match N feature vectors against the gallery. Return global IDs."""
        if feats.shape[0] == 0:
            return []

        ids: list[int] = []
        used_gids: set[int] = set()

        if self._gallery:
            gallery_ids = list(self._gallery.keys())
            gallery_feats = torch.stack([
                torch.stack(self._gallery[gid]).mean(dim=0)
                for gid in gallery_ids
            ])
            gallery_feats = F.normalize(gallery_feats, p=2, dim=-1)
            sim = torch.matmul(feats.cpu(), gallery_feats.T)  # (N, G)

            for i in range(feats.shape[0]):
                row = sim[i].clone()
                for gid_idx, gid in enumerate(gallery_ids):
                    if gid in used_gids:
                        row[gid_idx] = -2.0
                best_idx = row.argmax().item()
                best_sim = row[best_idx].item()

                if best_sim >= self._threshold:
                    gid = gallery_ids[best_idx]
                    used_gids.add(gid)
                else:
                    gid = self._next_id
                    self._next_id += 1

                ids.append(gid)
                self._gallery.setdefault(gid, [])
                self._gallery[gid].append(feats[i].cpu())
                if len(self._gallery[gid]) > self._window:
                    self._gallery[gid].pop(0)
        else:
            for i in range(feats.shape[0]):
                gid = self._next_id
                self._next_id += 1
                ids.append(gid)
                self._gallery[gid] = [feats[i].cpu()]

        return ids


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    ids: list[int],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Draw bounding boxes and ID labels on frame."""
    for (x1, y1, x2, y2), gid in zip(boxes, ids):
        color = _PALETTE[gid % len(_PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{gid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    info = f"Frame {frame_idx}  |  {len(ids)} persons  |  {fps:.1f} FPS"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def run_video_inference(
    checkpoint: Path,
    video_path: Path,
    output_path: Path,
    cam_id: int = 0,
    threshold: float = 0.5,
    conf: float = 0.5,
    device: torch.device = torch.device("cpu"),
    yolo_model: str = "yolov8n.pt",
) -> None:
    """Run detection + Re-ID on a video file and write annotated output."""
    model = load_model(checkpoint, device)
    transform = build_val_transform()
    detector = YOLO(yolo_model)
    tracker = SimpleReIDTracker(threshold=threshold)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (w, h))

    print(f"[Video] {video_path.name}  {w}x{h}  {src_fps:.1f}fps  {total} frames")
    print(f"[Video] output → {output_path}")

    frame_idx = 0
    import time
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Detect persons (YOLO class 0 = person)
        results = detector(frame, classes=[0], conf=conf, verbose=False)
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes else np.empty((0, 4), dtype=int)

        # Crop detections → PIL
        crops: list[Image.Image] = []
        boxes: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_bgr = frame[y1:y2, x1:x2]
            crops.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
            boxes.append((x1, y1, x2, y2))

        # Extract Re-ID features and match
        feats = extract_features(model, crops, cam_id, device, transform)
        ids = tracker.update(feats)

        # Draw and write
        elapsed = time.time() - t0
        fps = frame_idx / max(elapsed, 1e-6)
        frame = draw_detections(frame, boxes, ids, frame_idx, fps)
        writer.write(frame)

        if frame_idx % 100 == 0 or frame_idx == 1:
            print(f"  frame {frame_idx}/{total}  persons={len(crops)}  unique_ids={len(tracker._gallery)}  fps={fps:.1f}")

    cap.release()
    writer.release()
    print(f"\n[Video] done. {frame_idx} frames processed, {len(tracker._gallery)} unique identities")
    print(f"[Video] saved → {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video Re-ID: YOLOv8 detection + CLIP-ReID")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to stage2_best.pth")
    p.add_argument("--video", type=Path, required=True, help="Input video file (.mp4, .avi, etc.)")
    p.add_argument("--output", type=Path, default=Path("output/video_reid.mp4"), help="Output annotated video")
    p.add_argument("--cam-id", type=int, default=0, help="Camera ID for SIE (default: 0)")
    p.add_argument("--threshold", type=float, default=0.5, help="Cosine similarity threshold for Re-ID matching")
    p.add_argument("--conf", type=float, default=0.5, help="YOLO detection confidence threshold")
    p.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO model (default: yolov8n.pt)")
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detected)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[ReID] device={device}")

    run_video_inference(
        checkpoint=args.checkpoint,
        video_path=args.video,
        output_path=args.output,
        cam_id=args.cam_id,
        threshold=args.threshold,
        conf=args.conf,
        device=device,
        yolo_model=args.yolo_model,
    )
