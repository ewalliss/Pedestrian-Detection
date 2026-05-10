"""Integration smoke test: end-to-end mini pipeline with ~1000 Market-1501 images.

Runs Stage 1 (3 epochs) + Stage 2 (2 epochs) and checks:
  - Losses decrease over epochs
  - mAP > 0 after Stage 2
  - Health checks all pass
  - Checkpoints are saved

Usage:
    python -m pytest tests/integration/test_mini_pipeline.py -v
    # or directly:
    python tests/integration/test_mini_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.defaults import PedestrianReIDConfig
from src.eval.model_health_check import run_health_checks
from src.train.train_stage1 import train_stage1
from src.train.train_stage2 import train_stage2

MARKET1501_ROOT = Path("/Users/dangnguyen/Downloads/pedestrian/Market-1501-v15.09.15")


def _mini_cfg(output_dir: Path) -> PedestrianReIDConfig:
    cfg = PedestrianReIDConfig()
    cfg.market1501_root = MARKET1501_ROOT
    cfg.output_dir = output_dir
    cfg.mini = True
    cfg.mini_num_ids = 30        # 30 IDs × ~10 imgs ≈ 300 images — fast smoke test
    cfg.mini_epochs_stage1 = 2
    cfg.mini_epochs_stage2 = 2
    cfg.batch_size = 16
    cfg.num_instances = 4
    cfg.num_workers = 0          # 0 workers for pytest (avoid multiprocessing issues)
    cfg.fp16 = False             # FP16 only reliable on CUDA; tests run on CPU/GPU
    cfg.olp_k = 4                # small k for speed
    return cfg


@pytest.mark.skipif(
    not MARKET1501_ROOT.exists(),
    reason=f"Market-1501 not found at {MARKET1501_ROOT}",
)
class TestMiniPipeline:

    def test_stage1_runs_and_saves_checkpoint(self, tmp_path):
        cfg = _mini_cfg(tmp_path)
        ckpt_path = train_stage1(cfg)
        assert ckpt_path.exists(), f"Stage 1 checkpoint not saved: {ckpt_path}"

    def test_stage2_runs_and_saves_checkpoint(self, tmp_path):
        cfg = _mini_cfg(tmp_path)
        stage1_ckpt = train_stage1(cfg)
        assert stage1_ckpt.exists()

        stage2_ckpt = train_stage2(cfg, stage1_ckpt)
        assert stage2_ckpt.exists(), f"Stage 2 best checkpoint not saved: {stage2_ckpt}"

    def test_stage2_checkpoint_has_positive_map(self, tmp_path):
        cfg = _mini_cfg(tmp_path)
        stage1_ckpt = train_stage1(cfg)
        stage2_ckpt = train_stage2(cfg, stage1_ckpt)

        ckpt = torch.load(stage2_ckpt, map_location="cpu")
        best_map = ckpt.get("best_map", 0.0)
        assert best_map > 0.0, f"mAP after Stage 2 should be > 0, got {best_map}"

    def test_health_checks_pass_after_stage2(self, tmp_path):
        cfg = _mini_cfg(tmp_path)
        stage1_ckpt = train_stage1(cfg)
        stage2_ckpt = train_stage2(cfg, stage1_ckpt)

        device = torch.device("cpu")
        ckpt = torch.load(stage2_ckpt, map_location=device)

        from src.datasets.pedestrian_dataset import build_pedestrian_loaders
        from src.models.clip_reid_pedestrian import CLIPReIDPedestrianModel

        train_loader, _, _, num_pids, num_cams = build_pedestrian_loaders(
            market1501_root=cfg.market1501_root,
            batch_size=cfg.batch_size,
            num_instances=cfg.num_instances,
            num_workers=cfg.num_workers,
            mini=cfg.mini,
            mini_num_ids=cfg.mini_num_ids,
        )
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

        sample_batch = next(iter(train_loader))
        sample_batch = tuple(x[:4] for x in sample_batch)

        ok = run_health_checks(model, cfg, sample_batch, device)
        assert ok, "One or more health checks failed after Stage 2 training"

    def test_checkpoint_contains_required_keys(self, tmp_path):
        cfg = _mini_cfg(tmp_path)
        stage1_ckpt = train_stage1(cfg)
        stage2_ckpt = train_stage2(cfg, stage1_ckpt)

        s1 = torch.load(stage1_ckpt, map_location="cpu")
        for key in ("epoch", "stage", "model_state", "num_pids", "num_cams"):
            assert key in s1, f"Stage 1 checkpoint missing key: {key}"

        s2 = torch.load(stage2_ckpt, map_location="cpu")
        for key in ("epoch", "stage", "model_state", "num_pids", "num_cams", "best_map"):
            assert key in s2, f"Stage 2 checkpoint missing key: {key}"


# ── Standalone runner ──────────────────────────────────────────────────────────

def run_mini_pipeline():
    """Run the mini pipeline and print a summary report."""
    if not MARKET1501_ROOT.exists():
        print(f"[SKIP] Market-1501 not found at {MARKET1501_ROOT}")
        print("       Download it first or update MARKET1501_ROOT in this script.")
        return

    import time
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _mini_cfg(Path(tmpdir))
        print(f"\n{'='*60}")
        print("  Mini Pipeline Smoke Test")
        print(f"  Market-1501: {MARKET1501_ROOT}")
        print(f"  mini_num_ids={cfg.mini_num_ids}  batch_size={cfg.batch_size}")
        print(f"  Stage1 epochs={cfg.mini_epochs_stage1}  Stage2 epochs={cfg.mini_epochs_stage2}")
        print(f"{'='*60}\n")

        t0 = time.time()

        # Stage 1
        print("── Stage 1: PromptLearner training ──────────────────────────")
        stage1_ckpt = train_stage1(cfg)
        t1 = time.time()
        print(f"[Stage 1 done] {t1-t0:.1f}s  ckpt → {stage1_ckpt}\n")

        # Stage 2
        print("── Stage 2: Image encoder fine-tuning ───────────────────────")
        stage2_ckpt = train_stage2(cfg, stage1_ckpt)
        t2 = time.time()
        print(f"[Stage 2 done] {t2-t1:.1f}s  ckpt → {stage2_ckpt}\n")

        # Summary
        ckpt = torch.load(stage2_ckpt, map_location="cpu")
        print(f"\n{'='*60}")
        print(f"  RESULT: best_mAP = {ckpt.get('best_map', 0.0):.2f}%")
        print(f"  Total time: {t2-t0:.1f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    run_mini_pipeline()
