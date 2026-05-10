"""V-Track Live Re-ID Demo — video file or RTSP camera.

Reads frames from a video file or RTSP stream, detects person bounding boxes
with a Faster R-CNN (torchvision, pretrained on COCO), crops each person,
runs the Re-ID model, and overlays:

  - Coloured bounding box per person
    · Green  = matched a gallery identity (confidence ≥ threshold)
    · Red    = no gallery match (new or uncertain person)
  - ID label + cosine similarity score
  - Per-frame FPS and latency (detect ms + reid ms) in the top-left corner

Gallery
-------
The gallery is built from either:
  (a) A Market-1501 query/gallery split  (--market1501-root)
  (b) A folder of person crops           (--crops-dir)

In both cases features are pre-extracted once, then kept as a fixed gallery
that the live feed matches against.  A temporal feature bank (rolling average
over the last `--bank-window` frames) is used when the same person reappears.

Matching score
--------------
Score = cosine similarity of 1024-d BN features ∈ [-1, 1].
Recommended thresholds:
  ≥ 0.80  — high confidence same person
  0.60–0.79 — medium confidence
  < 0.60  — likely different person (assign new ID)

Threshold is configurable via --threshold (default 0.75).

Usage — video file
------------------
python scripts/demo_live.py \\
    --checkpoint model/stage2_best_c_loss.pth \\
    --source /path/to/video.mp4 \\
    --market1501-root /path/to/Market-1501-v15.09.15

Usage — RTSP camera
-------------------
python scripts/demo_live.py \\
    --checkpoint model/stage2_best_c_loss.pth \\
    --source rtsp://user:pass@192.168.1.100/stream1

Usage — webcam (device index)
------------------------------
python scripts/demo_live.py \\
    --checkpoint model/stage2_best_c_loss.pth \\
    --source 0

Flags
-----
--threshold T          Cosine similarity accept threshold (default 0.75)
--det-score T          Faster R-CNN person confidence threshold (default 0.70)
--bank-window N        Temporal feature bank rolling window (default 30)
--no-gallery           Skip pre-loading gallery; every person gets a new ID
--crops-dir PATH       Pre-cropped persons folder (subfolders = identity IDs)
--max-frames N         Stop after N frames (0 = run until end / q pressed)
--output PATH          Write annotated video to file instead of displaying
--no-fp16              Disable FP16
--det-batch N          Person crops per Re-ID batch (default 32)
--display-scale F      Scale factor for the display window (default 1.0)
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.datasets.transforms import build_val_transform
from src.eval.evaluate import power_normalize
from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False


# ── Visual constants ──────────────────────────────────────────────────────────

# BGR for OpenCV
_MATCHED   = (0, 210, 0)      # green
_UNMATCHED = (30, 30, 220)    # red
_LOW_CONF  = (0, 180, 255)    # orange — score between LOW and threshold
_OVERLAY   = (20, 20, 20)     # semi-transparent overlay bg

_FONT      = cv2.FONT_HERSHEY_SIMPLEX if _CV2_OK else None
_FONT_SCALE = 0.45
_FONT_THICK = 1


# ── Feature bank: rolling temporal average per identity ───────────────────────

class _FeatureBank:
    """Rolling-average feature bank per identity.

    For each tracked global identity, keep the last `window` embeddings and
    return their mean as the representative gallery vector.
    """

    def __init__(self, window: int = 30, embed_dim: int = 1024) -> None:
        self._window = window
        self._embed_dim = embed_dim
        # gid -> deque of (1024,) tensors on CPU
        self._bank: dict[int, collections.deque] = {}

    def update(self, gid: int, feat: torch.Tensor) -> None:
        if gid not in self._bank:
            self._bank[gid] = collections.deque(maxlen=self._window)
        self._bank[gid].append(feat.cpu())

    def gallery(self) -> tuple[list[int], torch.Tensor]:
        """Return (gids, gallery_feats) where gallery_feats is (K, 1024)."""
        if not self._bank:
            return [], torch.empty(0, self._embed_dim)
        gids = sorted(self._bank.keys())
        feats = torch.stack(
            [torch.stack(list(self._bank[g])).mean(0) for g in gids]
        )
        return gids, F.normalize(feats, p=2, dim=-1)


# ── Detector: Faster R-CNN person filter ──────────────────────────────────────

class _PersonDetector:
    """Wraps torchvision Faster R-CNN for person detection.

    Returns bounding boxes as (x1, y1, x2, y2) in pixel coordinates,
    filtered to class_id == 1 (person) with score ≥ det_score.
    """

    _PERSON_CLASS = 1  # COCO class 1 = person

    def __init__(self, device: torch.device, det_score: float = 0.70) -> None:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self._model.to(device).eval()
        self._preprocess = weights.transforms()
        self._device = device
        self._det_score = det_score

    @torch.no_grad()
    def detect(self, frame_bgr) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) person boxes in pixel coords."""
        # frame_bgr: numpy HxWx3 BGR
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_t = self._preprocess(img_pil).unsqueeze(0).to(self._device)
        pred = self._model(img_t)[0]

        boxes = []
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if label.item() == self._PERSON_CLASS and score.item() >= self._det_score:
                x1, y1, x2, y2 = box.int().tolist()
                boxes.append((x1, y1, x2, y2))
        return boxes


# ── Re-ID feature extractor (wraps model.extract_features) ────────────────────

class _ReIDExtractor:
    def __init__(
        self,
        model: CLIPReIDPedestrianModel,
        device: torch.device,
        fp16: bool,
        cam_id: int = 1,
        view_id: int = 0,
    ) -> None:
        self._model  = model
        self._device = device
        self._fp16   = fp16 and device.type == "cuda"
        self._tfm    = build_val_transform()
        # Fixed cam/view IDs for live input (single camera)
        self._cam_id  = torch.tensor([cam_id],  dtype=torch.long, device=device)
        self._view_id = torch.tensor([view_id], dtype=torch.long, device=device)

    @torch.no_grad()
    def extract_batch(self, crops_pil: list[Image.Image]) -> torch.Tensor:
        """Return (N, 1024) normalised feature tensor for a list of PIL crops."""
        if not crops_pil:
            return torch.empty(0, 1024, device="cpu")
        imgs = torch.stack([self._tfm(c) for c in crops_pil]).to(self._device)
        cam_ids  = self._cam_id.expand(len(crops_pil))
        view_ids = self._view_id.expand(len(crops_pil))
        with torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=self._fp16,
        ):
            feats = self._model.extract_features(imgs, cam_ids, view_ids)
        feats = F.normalize(power_normalize(feats.float(), 0.5), p=2, dim=-1)
        return feats.cpu()


# ── Gallery builder ───────────────────────────────────────────────────────────

def _build_gallery_from_market(
    root: Path,
    model: CLIPReIDPedestrianModel,
    device: torch.device,
    fp16: bool,
    split: str = "gallery",
    batch_size: int = 128,
) -> tuple[list[int], torch.Tensor]:
    """Pre-extract Market-1501 gallery features. Returns (pids, feats)."""
    from src.datasets.market1501 import Market1501
    from torch.utils.data import DataLoader

    tfm = build_val_transform()
    ds  = Market1501(root, split=split, transform=tfm, remap_pids=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    feats, pids = [], []
    model.eval()
    print(f"[gallery] extracting {len(ds):,} images from Market-1501 {split}…")
    for imgs, pid_batch, cam_ids, view_ids in loader:
        imgs     = imgs.to(device)
        cam_ids  = cam_ids.to(device)
        view_ids = view_ids.to(device)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=fp16 and device.type == "cuda"
        ):
            f = model.extract_features(imgs, cam_ids, view_ids)
        feats.append(F.normalize(power_normalize(f.float().cpu(), 0.5), p=2, dim=-1))
        pids.extend(pid_batch.tolist())

    feats_t = F.normalize(torch.cat(feats, dim=0), p=2, dim=-1)
    print(f"[gallery] done. feats={list(feats_t.shape)}")
    return pids, feats_t


def _build_gallery_from_crops(
    crops_dir: Path,
    model: CLIPReIDPedestrianModel,
    device: torch.device,
    fp16: bool,
) -> tuple[list[int], torch.Tensor]:
    """Build gallery from a folder: crops_dir/{pid}/*.jpg."""
    tfm = build_val_transform()
    entries = []
    for pid_dir in sorted(crops_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        try:
            pid = int(pid_dir.name)
        except ValueError:
            continue
        for img_path in sorted(pid_dir.glob("*.jpg")):
            entries.append((pid, img_path))

    if not entries:
        raise RuntimeError(f"No crops found in {crops_dir}")

    feats, pids = [], []
    model.eval()
    cam_id  = torch.tensor([1], dtype=torch.long, device=device)
    view_id = torch.tensor([0], dtype=torch.long, device=device)

    print(f"[gallery] extracting {len(entries):,} crop images from {crops_dir}…")
    for pid, img_path in entries:
        img = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=fp16 and device.type == "cuda"
        ):
            f = model.extract_features(img, cam_id, view_id)
        feats.append(F.normalize(power_normalize(f.float().cpu(), 0.5), p=2, dim=-1))
        pids.append(pid)

    feats_t = F.normalize(torch.cat(feats, dim=0), p=2, dim=-1)
    print(f"[gallery] done. feats={list(feats_t.shape)}")
    return pids, feats_t


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _color_for_score(score: float, threshold: float) -> tuple[int, int, int]:
    if score >= threshold:
        return _MATCHED
    if score >= threshold - 0.15:
        return _LOW_CONF
    return _UNMATCHED


def _draw_box(
    frame,
    box: tuple[int, int, int, int],
    gid: int,
    score: float,
    threshold: float,
) -> None:
    x1, y1, x2, y2 = box
    color = _color_for_score(score, threshold)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label  = f"ID:{gid}  {score:.2f}" if score >= 0 else f"ID:{gid} new"
    (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
    ty = max(y1 - 4, th + 4)
    # label background
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, cv2.FILLED)
    cv2.putText(
        frame, label, (x1 + 2, ty - 2),
        _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK, cv2.LINE_AA,
    )


def _draw_hud(
    frame,
    fps: float,
    det_ms: float,
    reid_ms: float,
    n_persons: int,
    threshold: float,
) -> None:
    lines = [
        f"FPS: {fps:.1f}   det: {det_ms:.1f}ms  reid: {reid_ms:.1f}ms",
        f"persons: {n_persons}   threshold: ≥{threshold:.2f}",
        "Green=matched  Orange=low-conf  Red=unmatched",
    ]
    y = 20
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.5, 1)
        cv2.rectangle(frame, (8, y - th - 4), (12 + tw, y + 4), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, line, (10, y), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += th + 10


# ── Source normalisation ──────────────────────────────────────────────────────

def _open_source(source: str):
    """Open cv2.VideoCapture from file path, RTSP URL, or integer device."""
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return cap


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _CV2_OK:
        print("[error] OpenCV not installed. Install with:  pip install opencv-python")
        sys.exit(1)

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
    print(f"[demo] device={device}  fp16={fp16}  source={args.source}")

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_pids = ckpt.get("num_pids", 751)
    num_cams = ckpt.get("num_cams", 30)
    print(f"[model] num_pids={num_pids}  num_cams={num_cams}  epoch={ckpt.get('epoch','?')}")
    model = CLIPReIDPedestrianModel(num_pids=num_pids, num_cams=num_cams)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    # ── Detector ──────────────────────────────────────────────────────────────
    print("[detector] loading Faster R-CNN ResNet50-FPN v2…")
    detector = _PersonDetector(device, det_score=args.det_score)

    # ── Gallery ───────────────────────────────────────────────────────────────
    gallery_pids: list[int] = []
    gallery_feats: torch.Tensor = torch.empty(0, 1024)

    if not args.no_gallery:
        if args.crops_dir:
            gallery_pids, gallery_feats = _build_gallery_from_crops(
                Path(args.crops_dir), model, device, fp16
            )
        elif args.market1501_root:
            gallery_pids, gallery_feats = _build_gallery_from_market(
                Path(args.market1501_root), model, device, fp16,
                split="gallery"
            )
        print(f"[gallery] {len(gallery_pids):,} entries  feat_dim={gallery_feats.shape[1]}")

    # ── Live components ────────────────────────────────────────────────────────
    extractor = _ReIDExtractor(
        model, device, fp16,
        cam_id=args.cam_id, view_id=0,
    )
    bank = _FeatureBank(window=args.bank_window)
    local_to_gid: dict[int, int] = {}  # track_id -> global id (simple counter)
    gid_counter = 0

    # ── Video source ──────────────────────────────────────────────────────────
    cap = _open_source(args.source)
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[source] fps={fps_native:.1f}  total_frames={total_frames or '?'}")

    # ── Output writer (optional) ───────────────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps_native, (w, h))
        print(f"[output] writing to {args.output}")

    frame_idx  = 0
    fps_window = collections.deque(maxlen=30)
    t_frame    = time.perf_counter()

    print("\n[running] Press 'q' to quit.\n")
    print(f"  Matching score : cosine similarity ∈ [-1, 1]")
    print(f"  Threshold      : ≥ {args.threshold:.2f}  (configurable via --threshold)")
    print(f"    ≥ 0.80  — high confidence same person  → green")
    print(f"    0.60–{args.threshold:.2f}  — low confidence             → orange")
    print(f"    < 0.60  — unmatched / new person       → red\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        # ── Detection ─────────────────────────────────────────────────────────
        t_det_start = time.perf_counter()
        boxes = detector.detect(frame)
        det_ms = (time.perf_counter() - t_det_start) * 1000

        # ── Crop + Re-ID ──────────────────────────────────────────────────────
        reid_ms = 0.0
        if boxes:
            h_frame, w_frame = frame.shape[:2]
            crops_pil = []
            for x1, y1, x2, y2 in boxes:
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w_frame, x2); y2 = min(h_frame, y2)
                if x2 <= x1 or y2 <= y1:
                    crops_pil.append(Image.new("RGB", (64, 128)))
                    continue
                crop_bgr = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crops_pil.append(Image.fromarray(crop_rgb))

            t_reid_start = time.perf_counter()
            # Extract features in configurable batches
            all_feats_list = []
            for i in range(0, len(crops_pil), args.det_batch):
                batch = crops_pil[i : i + args.det_batch]
                all_feats_list.append(extractor.extract_batch(batch))
            query_feats = torch.cat(all_feats_list, dim=0)  # (N, 1024)
            reid_ms = (time.perf_counter() - t_reid_start) * 1000

            # ── Match against combined gallery: static + temporal bank ────────
            bank_gids, bank_feats = bank.gallery()
            all_gids   = gallery_pids + bank_gids
            all_g_feats = torch.cat([gallery_feats, bank_feats], dim=0) \
                          if gallery_feats.shape[0] > 0 and bank_feats.shape[0] > 0 \
                          else (gallery_feats if gallery_feats.shape[0] > 0 else bank_feats)

            for i, (box, q_feat) in enumerate(zip(boxes, query_feats)):
                # Local track_id (simple box index per frame — no actual tracker)
                # For a real deployment wire this to ByteTrack / SORT output.
                track_id = i

                if all_g_feats.shape[0] > 0:
                    scores = torch.matmul(q_feat.unsqueeze(0), all_g_feats.T).squeeze(0)
                    best_score, best_idx = scores.max(0)
                    best_score = best_score.item()
                else:
                    best_score, best_idx = -1.0, -1

                if best_score >= args.threshold:
                    gid = all_gids[best_idx]
                    local_to_gid[track_id] = gid
                else:
                    # Assign new global ID
                    if track_id not in local_to_gid:
                        gid_counter += 1
                        local_to_gid[track_id] = gid_counter
                    gid = local_to_gid[track_id]
                    best_score = -1.0

                bank.update(gid, q_feat)
                _draw_box(frame, box, gid, best_score, args.threshold)

        # ── FPS HUD ───────────────────────────────────────────────────────────
        now = time.perf_counter()
        fps_window.append(1.0 / max(now - t_frame, 1e-9))
        t_frame = now
        fps = sum(fps_window) / len(fps_window)
        _draw_hud(frame, fps, det_ms, reid_ms, len(boxes), args.threshold)

        # ── Display ───────────────────────────────────────────────────────────
        if writer:
            writer.write(frame)

        if not args.output:
            display = frame
            if args.display_scale != 1.0:
                dh = int(frame.shape[0] * args.display_scale)
                dw = int(frame.shape[1] * args.display_scale)
                display = cv2.resize(frame, (dw, dh))
            cv2.imshow("V-Track Live Re-ID", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if not args.output:
        cv2.destroyAllWindows()

    print(f"\n[done] processed {frame_idx} frames")
    if fps_window:
        print(f"  avg FPS: {sum(fps_window)/len(fps_window):.1f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V-Track live Re-ID demo — video or RTSP")
    p.add_argument("--checkpoint", "--ckpt", type=str,
                   default="model/stage2_best_c_loss.pth")
    p.add_argument("--source", type=str, required=True,
                   help="Video file path, RTSP URL, or webcam index (e.g. 0)")
    p.add_argument("--market1501-root", type=str, default=None,
                   help="Market-1501 root — pre-load as static gallery")
    p.add_argument("--crops-dir", type=str, default=None,
                   help="Folder of pre-cropped persons ({pid}/*.jpg) as gallery")
    p.add_argument("--no-gallery", action="store_true",
                   help="Skip pre-loading gallery; assign new IDs to everyone")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Cosine similarity accept threshold (default 0.75)")
    p.add_argument("--det-score", type=float, default=0.70,
                   help="Faster R-CNN person confidence threshold (default 0.70)")
    p.add_argument("--bank-window", type=int, default=30,
                   help="Temporal feature bank rolling window size")
    p.add_argument("--cam-id", type=int, default=1,
                   help="Camera ID passed to SIE (1-indexed; default 1)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited)")
    p.add_argument("--output", type=str, default=None,
                   help="Write annotated video to file (mp4)")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--det-batch", type=int, default=32,
                   help="Re-ID inference batch size for crops (default 32)")
    p.add_argument("--display-scale", type=float, default=1.0,
                   help="Scale factor for display window (default 1.0)")
    return p.parse_args()


if __name__ == "__main__":
    main()
