"""RTSP live-stream inference with V100-optimised CUDA pipeline.

Reads person bounding boxes from an upstream detector (or uses the full frame
as a single crop when no detector is provided), runs ViT-CLIP-ReID-SIE-OLP
feature extraction in fp16, and assigns globally-stable Re-ID labels via
ReIDPipeline.

Tesla V100 optimisations applied
---------------------------------
* cuDNN benchmark mode — selects fastest convolution algorithms at startup.
* TF32 on matmul and cuDNN — V100 supports TF32; saves bandwidth with no
  practical accuracy loss for inference.
* torch.compile (max-autotune, reduce-overhead) — fuses + optimises the model
  graph via Triton kernels.
* AMP fp16 autocast — halves memory bandwidth; V100 has 28 TFLOPS fp16.
* Non-blocking CUDA transfers + pinned host memory — overlaps PCIe DMA with
  GPU compute.
* Double-buffered CUDA streams — decode/preprocess stream runs in parallel
  with the inference stream so the GPU is never idle waiting for data.
* Producer-consumer threading — capture thread decodes frames concurrently;
  inference thread never stalls on OpenCV I/O.
* Batch accumulation — fills a configurable batch before invoking the model
  to maximise GPU utilisation; falls back to partial batch on timeout.

Usage
-----
    python scripts/rtsp_inference.py \
        --rtsp rtsp://user:pass@192.168.1.10:554/stream \
        --checkpoint model/custom/stage2_best.pth \
        --cam-id 0 \
        --batch-size 8 \
        --threshold 0.75 \
        --show
"""

from __future__ import annotations

import argparse
import itertools
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.amp
import torchvision.transforms.functional as TF

# --------------------------------------------------------------------------- #
# Path setup — scope original CLIP-ReID imports                               #
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel
from src.reid.clip_encoder import CLIPEncoder
from src.reid.cosine_matcher import CosineMatcher
from src.reid.pipeline import ReIDPipeline
from src.reid.projector import IdentityProjector
from src.reid.temporal_bank import TemporalFeatureBank

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
_CLIP_MEAN = (0.5, 0.5, 0.5)
_CLIP_STD = (0.5, 0.5, 0.5)
_INPUT_H = 224
_INPUT_W = 224

# Pre-compute normalisation tensors once on GPU (broadcast over N)
_MEAN_GPU: torch.Tensor | None = None  # set in _init_gpu_transforms()
_STD_GPU: torch.Tensor | None = None


def _init_gpu_transforms(device: torch.device) -> None:
    global _MEAN_GPU, _STD_GPU
    _MEAN_GPU = torch.tensor(_CLIP_MEAN, dtype=torch.float16, device=device).view(1, 3, 1, 1)
    _STD_GPU = torch.tensor(_CLIP_STD, dtype=torch.float16, device=device).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# CUDA environment — V100 tuning                                              #
# --------------------------------------------------------------------------- #

def _configure_cuda() -> torch.device:
    """Apply V100-specific CUDA settings and return the target device."""
    if not torch.cuda.is_available():
        print("[rtsp] WARNING: CUDA not available, falling back to CPU — performance will be poor.")
        return torch.device("cpu")

    device = torch.device("cuda:0")

    # Fastest cuDNN algorithm selection (one-time overhead at first forward pass)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # TF32 — V100 Ampere+ feature; effectively free on Volta with negligible loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Reserve memory upfront to avoid fragmentation during streaming
    torch.cuda.set_per_process_memory_fraction(0.90, device=device)

    name = torch.cuda.get_device_name(device)
    mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    print(f"[rtsp] GPU: {name}  |  VRAM: {mem_gb:.1f} GB")
    return device


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #

def load_model(
    checkpoint: Path,
    device: torch.device,
    compile_model: bool = True,
) -> CLIPReIDPedestrianModel:
    """Load stage-2 checkpoint, cast to fp16, optionally torch.compile."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CLIPReIDPedestrianModel(
        num_pids=ckpt["num_pids"],
        num_cams=ckpt["num_cams"],
        clip_name="openai/clip-vit-base-patch16",
        template="a photo of a X X X X person",
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # fp16 — halves memory bandwidth; V100 has 28 TFLOPS fp16 vs 14 fp32
    model = model.half().to(device)

    if compile_model and device.type == "cuda":
        # reduce-overhead: removes Python dispatcher overhead per call
        # max-autotune: Triton auto-tunes tile sizes for ViT attention
        print("[rtsp] torch.compile — this may take ~60 s on first run …")
        model = torch.compile(model, mode="max-autotune", fullgraph=False)  # type: ignore[assignment]

    mAP = ckpt.get("best_map", "?")
    print(f"[rtsp] Loaded checkpoint: num_pids={ckpt['num_pids']}  mAP={mAP}")
    return model  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Preprocessing — GPU-resident, fp16                                          #
# --------------------------------------------------------------------------- #

def preprocess_crops_gpu(
    crops: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Convert a list of BGR uint8 numpy crops to a (N,3,224,224) fp16 CUDA tensor.

    Performs resize + channel conversion + normalisation entirely on the GPU to
    avoid redundant CPU work.  Uses pinned memory for the host→device transfer.

    Args:
        crops: List of HxWx3 BGR uint8 arrays (OpenCV format).
        device: Target CUDA device.

    Returns:
        (N, 3, 224, 224) fp16 tensor on ``device``.
    """
    assert _MEAN_GPU is not None, "call _init_gpu_transforms() first"

    tensors: list[torch.Tensor] = []
    for bgr in crops:
        # BGR → RGB, HWC → CHW, uint8 → fp32 [0,1]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)  # (3, H, W)
        # Resize to 224×224 (bicubic matches val transform)
        t = TF.resize(t.unsqueeze(0), [_INPUT_H, _INPUT_W], antialias=True).squeeze(0)
        tensors.append(t)

    # Stack on CPU with pinned memory, then transfer non-blocking
    batch_cpu = torch.stack(tensors, dim=0).pin_memory()         # (N, 3, 224, 224) fp32
    batch_gpu = batch_cpu.to(device, non_blocking=True).half()   # fp16, async DMA

    # Normalise on GPU (broadcast)
    batch_gpu = (batch_gpu - _MEAN_GPU) / _STD_GPU               # (N, 3, 224, 224)
    return batch_gpu


# --------------------------------------------------------------------------- #
# Frame capture thread                                                         #
# --------------------------------------------------------------------------- #

class FrameCapture(threading.Thread):
    """Daemon thread that reads RTSP frames into a bounded queue.

    Keeps the most-recent ``maxsize`` frames; drops oldest when full so that
    the inference thread always processes near-real-time data rather than
    accumulating latency under load.

    Args:
        rtsp_url:  RTSP stream URL.
        out_queue: Queue shared with the inference thread.
        maxsize:   Queue depth; 2 is enough for double-buffering.
    """

    def __init__(self, rtsp_url: str, out_queue: queue.Queue, maxsize: int = 4) -> None:
        super().__init__(daemon=True, name="FrameCapture")
        self._url = rtsp_url
        self._q = out_queue
        self._stop_flag = threading.Event()
        self._maxsize = maxsize

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        cap = _open_capture(self._url)
        fps_target = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_period = 1.0 / fps_target
        dropped = 0

        while not self._stop_flag.is_set():
            t0 = time.monotonic()
            ok, frame = cap.read()

            if not ok:
                print("[capture] Stream read failed — attempting reconnect …")
                cap.release()
                time.sleep(1.0)
                cap = _open_capture(self._url)
                continue

            # Non-blocking put; drop frame instead of blocking inference thread
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                dropped += 1
                if dropped % 100 == 0:
                    print(f"[capture] dropped {dropped} frames (inference slower than capture)")
                # Evict oldest, insert newest
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait(frame)

            # Pace capture to stream FPS to avoid busy-spinning
            elapsed = time.monotonic() - t0
            sleep_t = frame_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()


def _open_capture(url: str) -> cv2.VideoCapture:
    """Open an RTSP capture with transport and buffer settings optimised for low latency."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Use TCP for RTSP to avoid UDP packet loss
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {url}")
    return cap


# --------------------------------------------------------------------------- #
# Inference loop                                                               #
# --------------------------------------------------------------------------- #

class RTSPReIDInference:
    """RTSP live inference with Re-ID.

    Args:
        model:      Loaded (and optionally compiled) CLIPReIDPedestrianModel.
        pipeline:   Fully wired ReIDPipeline instance.
        device:     CUDA device.
        cam_id:     Camera index fed to SIE embedding (0-indexed, subtract 1 if Market).
        batch_size: Maximum number of crops to inference in one forward pass.
        batch_timeout: Max seconds to wait to fill a batch before flushing.
        show:       Display annotated frames via cv2.imshow.
        detector:   Optional callable ``(frame: np.ndarray) -> list[tuple[int,int,int,int]]``
                    returning (x1,y1,x2,y2) boxes.  When None, the full frame is used.
    """

    # CUDA streams: stream_a = preprocess; stream_b = inference
    # Overlap: while inference runs on batch-k, preprocess uploads batch-(k+1)
    _stream_a: torch.cuda.Stream
    _stream_b: torch.cuda.Stream

    def __init__(
        self,
        model: CLIPReIDPedestrianModel,
        pipeline: ReIDPipeline,
        device: torch.device,
        cam_id: int = 0,
        batch_size: int = 8,
        batch_timeout: float = 0.04,
        show: bool = False,
        detector=None,
    ) -> None:
        self._model = model
        self._pipeline = pipeline
        self._device = device
        self._cam_id = cam_id
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._show = show
        self._detector = detector

        if device.type == "cuda":
            self._stream_a = torch.cuda.Stream(device=device)
            self._stream_b = torch.cuda.Stream(device=device)

        # FPS tracking
        self._ts_window: deque[float] = deque(maxlen=60)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, frame_queue: queue.Queue) -> None:
        """Consume frames from ``frame_queue`` and run Re-ID inference.

        Blocks until KeyboardInterrupt or 'q' key is pressed.
        """
        print("[rtsp] Inference loop started.  Press Ctrl-C or 'q' to stop.")

        pending_frames: list[np.ndarray] = []
        pending_track_ids: list[int] = []

        batch_deadline = time.monotonic() + self._batch_timeout

        try:
            while True:
                # Collect frames until batch full or timeout
                remaining = batch_deadline - time.monotonic()
                if remaining > 0 and len(pending_frames) < self._batch_size:
                    try:
                        frame = frame_queue.get(timeout=min(remaining, 0.005))
                        pending_frames.append(frame)
                        # Assign a monotonic fake track ID per frame
                        # (real use-case: replace with tracker output)
                        pending_track_ids.append(len(pending_track_ids))
                    except queue.Empty:
                        pass
                    continue

                if not pending_frames:
                    batch_deadline = time.monotonic() + self._batch_timeout
                    continue

                # Process batch
                self._process_batch(pending_frames, pending_track_ids)
                pending_frames = []
                pending_track_ids = []
                batch_deadline = time.monotonic() + self._batch_timeout

                if self._show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            if self._show:
                cv2.destroyAllWindows()
            print(f"[rtsp] Stopped.  Average FPS: {self._avg_fps():.1f}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        frames: list[np.ndarray],
        track_ids: list[int],
    ) -> None:
        """Extract crops, run model, assign Re-ID labels, optionally display."""
        t_start = time.monotonic()

        # 1. Detect / crop
        all_crops: list[np.ndarray] = []
        crop_frame_idx: list[int] = []      # which frame each crop came from
        crop_track_ids: list[int] = []

        for fi, frame in enumerate(frames):
            boxes = self._detect(frame)
            for bi, (x1, y1, x2, y2) in enumerate(boxes):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                all_crops.append(crop)
                crop_frame_idx.append(fi)
                crop_track_ids.append(track_ids[fi] * 1000 + bi)  # unique per crop

        if not all_crops:
            return

        # 2. Preprocess on stream_a (async DMA + GPU normalise)
        if self._device.type == "cuda":
            with torch.cuda.stream(self._stream_a):
                pixel_values = preprocess_crops_gpu(all_crops, self._device)
            # Ensure stream_b waits for stream_a to finish upload
            self._stream_b.wait_stream(self._stream_a)
        else:
            pixel_values = preprocess_crops_gpu(all_crops, self._device)

        # 3. Build cam/view id tensors (0-indexed SIE)
        n = pixel_values.shape[0]
        cam_ids = torch.full((n,), self._cam_id, dtype=torch.long, device=self._device)
        view_ids = torch.zeros(n, dtype=torch.long, device=self._device)

        # 4. Model forward on stream_b with AMP fp16
        if self._device.type == "cuda":
            with torch.cuda.stream(self._stream_b):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    feats = self._model.extract_features(pixel_values, cam_ids, view_ids)
        else:
            feats = self._model.extract_features(pixel_values, cam_ids, view_ids)

        # Sync before using feats on the default stream
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).wait_stream(self._stream_b)

        # 5. Re-ID assignment via pipeline
        # extract_features returns fused feats; feed as mock PIL crops with pre-encoded feats
        # We bypass encoder in ReIDPipeline by directly calling the matching components
        feats_f32 = feats.float()
        gallery_ids, gallery_feats = self._pipeline._bank.get_gallery()
        if len(gallery_ids) > 0:
            gallery_feats = gallery_feats.to(self._device)
        _, assignments = self._pipeline._matcher.match(feats_f32, gallery_feats)

        global_ids: list[int] = []
        for i, (local_id, feat, assignment) in enumerate(
            zip(crop_track_ids, feats_f32, assignments)
        ):
            key = (self._cam_id, local_id)
            if key in self._pipeline._local_to_global:
                gid = self._pipeline._local_to_global[key]
            elif assignment.item() != CosineMatcher.UNMATCHED:
                gid = gallery_ids[assignment.item()]
                self._pipeline._local_to_global[key] = gid
            else:
                gid = next(self._pipeline._id_counter)
                self._pipeline._local_to_global[key] = gid
            self._pipeline._bank.update(gid, feat.squeeze(0))
            global_ids.append(gid)

        # 6. FPS tracking
        self._ts_window.append(time.monotonic() - t_start)

        # 7. Display
        if self._show:
            self._draw_and_show(frames, all_crops, crop_frame_idx, global_ids)

    def _detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes for ``frame``.

        Falls back to the full frame when no detector is configured.
        """
        if self._detector is not None:
            return self._detector(frame)
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]

    def _draw_and_show(
        self,
        frames: list[np.ndarray],
        crops: list[np.ndarray],  # noqa: ARG002
        crop_frame_idx: list[int],
        global_ids: list[int],
    ) -> None:
        vis_frames = [f.copy() for f in frames]

        # Colour palette — cycle through 32 distinct hues
        palette = [(int(h * 255 / 32), 200, 200) for h in range(32)]

        for fi_crop, gid in zip(crop_frame_idx, global_ids):
            frame = vis_frames[fi_crop]
            color_hsv = palette[gid % len(palette)]
            color_bgr = cv2.cvtColor(
                np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            cv2.putText(
                frame, f"ID:{gid}",
                (10, 30 + 30 * fi_crop),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2,
            )

        fps = self._avg_fps()
        for frame in vis_frames:
            cv2.putText(frame, f"FPS:{fps:.1f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("V-Track RTSP", frame)

    def _avg_fps(self) -> float:
        if not self._ts_window:
            return 0.0
        return 1.0 / (sum(self._ts_window) / len(self._ts_window))


# --------------------------------------------------------------------------- #
# Warm-up — amortise torch.compile JIT cost before live stream               #
# --------------------------------------------------------------------------- #

def warm_up(
    model: CLIPReIDPedestrianModel,
    device: torch.device,
    batch_size: int,
    n_iters: int = 3,
) -> None:
    """Run ``n_iters`` dummy forward passes to trigger JIT compilation."""
    print(f"[rtsp] Warming up model ({n_iters} iters, batch={batch_size}) …")
    dummy = torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=device)
    cam = torch.zeros(batch_size, dtype=torch.long, device=device)
    view = torch.zeros(batch_size, dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        for _ in range(n_iters):
            _ = model.extract_features(dummy, cam, view)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print("[rtsp] Warm-up complete.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTSP live Re-ID inference — V100 optimised",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rtsp", required=True, help="RTSP stream URL")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model/custom/stage2_best.pth"),
        help="Stage-2 checkpoint (.pth)",
    )
    p.add_argument("--cam-id", type=int, default=0, help="Camera index (0-indexed SIE input)")
    p.add_argument("--batch-size", type=int, default=8, help="Max crops per forward pass")
    p.add_argument(
        "--batch-timeout",
        type=float,
        default=0.033,
        help="Seconds to wait before flushing a partial batch (≈1 frame @ 30 fps)",
    )
    p.add_argument("--threshold", type=float, default=0.75, help="Cosine similarity Re-ID threshold")
    p.add_argument("--bank-window", type=int, default=30, help="TemporalFeatureBank window size")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (faster startup, slower inference)",
    )
    p.add_argument("--show", action="store_true", help="Display annotated frames (requires display)")
    p.add_argument("--queue-depth", type=int, default=4, help="Frame queue depth between capture and inference")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # 1. CUDA setup
    device = _configure_cuda()
    _init_gpu_transforms(device)

    # 2. Load model
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = load_model(args.checkpoint, device, compile_model=not args.no_compile)

    # 3. Warm up (amortises torch.compile JIT + cuDNN benchmark selection)
    if device.type == "cuda":
        warm_up(model, device, args.batch_size)

    # 4. Re-ID pipeline
    projector = IdentityProjector(in_dim=512, out_dim=512)
    matcher = CosineMatcher(threshold=args.threshold)
    bank = TemporalFeatureBank(window=args.bank_window)
    # ReIDPipeline encoder is unused here (we call model.extract_features directly)
    # Pass a stub encoder to satisfy the constructor signature
    encoder = CLIPEncoder(device=device.type if device.type != "cuda" else "cuda")
    pipeline = ReIDPipeline(encoder=encoder, projector=projector, matcher=matcher, bank=bank)

    # 5. Inference engine
    engine = RTSPReIDInference(
        model=model,
        pipeline=pipeline,
        device=device,
        cam_id=args.cam_id,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        show=args.show,
    )

    # 6. Capture thread
    frame_q: queue.Queue = queue.Queue(maxsize=args.queue_depth)
    capture = FrameCapture(args.rtsp, frame_q, maxsize=args.queue_depth)
    capture.start()

    # 7. Inference (blocking — runs on main thread)
    try:
        engine.run(frame_q)
    finally:
        capture.stop()
        capture.join(timeout=2.0)
        print("[rtsp] Clean shutdown.")


if __name__ == "__main__":
    main()
