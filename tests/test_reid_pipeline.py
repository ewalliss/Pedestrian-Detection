"""Unit and integration tests for the Re-ID pipeline.

Run with:
    python -m pytest tests/test_reid_pipeline.py -v
"""

from __future__ import annotations

import torch
import pytest
from PIL import Image

from src.reid import (
    CLIPEncoder,
    CosineMatcher,
    IdentityProjector,
    ReIDPipeline,
    TemporalFeatureBank,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def encoder() -> CLIPEncoder:
    """Shared CLIP encoder (loaded once per test session to save time)."""
    return CLIPEncoder(device="cpu")


@pytest.fixture
def dummy_crop() -> Image.Image:
    return Image.new("RGB", (128, 256), color=(100, 150, 200))


@pytest.fixture
def dummy_crop_alt() -> Image.Image:
    """Visually very different crop (solid red) — should differ in embedding."""
    return Image.new("RGB", (128, 256), color=(200, 50, 10))


@pytest.fixture
def pipeline(encoder: CLIPEncoder) -> ReIDPipeline:
    return ReIDPipeline(
        encoder=encoder,
        projector=IdentityProjector(),
        matcher=CosineMatcher(threshold=0.75),
        bank=TemporalFeatureBank(window=30),
    )


# ---------------------------------------------------------------------------
# CLIPEncoder tests
# ---------------------------------------------------------------------------

class TestCLIPEncoder:
    def test_encode_output_shape(self, encoder: CLIPEncoder, dummy_crop: Image.Image):
        feats = encoder.encode([dummy_crop, dummy_crop])
        assert feats.shape == (2, 512), f"unexpected shape: {feats.shape}"

    def test_encode_l2_normalised(self, encoder: CLIPEncoder, dummy_crop: Image.Image):
        feats = encoder.encode([dummy_crop])
        norms = feats.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_empty_returns_empty(self, encoder: CLIPEncoder):
        feats = encoder.encode([])
        assert feats.shape == (0, 512)

    def test_first_layer_frozen(self, encoder: CLIPEncoder):
        first_layer = encoder.model.vision_model.encoder.layers[0]
        for name, param in first_layer.named_parameters():
            assert not param.requires_grad, (
                f"Layer 0 param '{name}' should be frozen but requires_grad=True"
            )

    def test_later_layers_trainable(self, encoder: CLIPEncoder):
        later_layer = encoder.model.vision_model.encoder.layers[1]
        trainable = [p for p in later_layer.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Layer 1+ should have trainable parameters"


# ---------------------------------------------------------------------------
# CosineMatcher tests
# ---------------------------------------------------------------------------

class TestCosineMatcher:
    def test_self_similarity_is_one(self):
        matcher = CosineMatcher(threshold=0.75)
        v = torch.randn(1, 512)
        v = v / v.norm(dim=-1, keepdim=True)
        _, assignments = matcher.match(v, v)
        assert assignments[0].item() == 0

    def test_below_threshold_unmatched(self):
        matcher = CosineMatcher(threshold=0.99)
        q = torch.tensor([[1.0, 0.0]])
        g = torch.tensor([[0.0, 1.0]])  # orthogonal -> sim = 0
        _, assignments = matcher.match(q, g)
        assert assignments[0].item() == CosineMatcher.UNMATCHED

    def test_empty_gallery_all_unmatched(self):
        matcher = CosineMatcher(threshold=0.75)
        q = torch.randn(3, 512)
        q = q / q.norm(dim=-1, keepdim=True)
        _, assignments = matcher.match(q, torch.zeros(0, 512))
        assert (assignments == CosineMatcher.UNMATCHED).all()

    def test_sim_matrix_shape(self):
        matcher = CosineMatcher(threshold=0.75)
        q = torch.randn(4, 512)
        g = torch.randn(6, 512)
        # normalise
        q = q / q.norm(dim=-1, keepdim=True)
        g = g / g.norm(dim=-1, keepdim=True)
        sim, _ = matcher.match(q, g)
        assert sim.shape == (4, 6)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CosineMatcher(threshold=0.0)


# ---------------------------------------------------------------------------
# TemporalFeatureBank tests
# ---------------------------------------------------------------------------

class TestTemporalFeatureBank:
    def test_window_eviction(self):
        bank = TemporalFeatureBank(window=30)
        feat = torch.randn(512)
        for _ in range(35):
            bank.update(identity_id=0, feature=feat)
        assert len(bank._bank[0]) == 30

    def test_mean_of_identical_features(self):
        bank = TemporalFeatureBank(window=30)
        v = torch.randn(512)
        v = v / v.norm()
        for _ in range(5):
            bank.update(identity_id=0, feature=v)
        _, gallery = bank.get_gallery()
        # mean of identical normalised vectors = the vector itself (after renorm)
        assert torch.allclose(gallery[0], v, atol=1e-5)

    def test_get_gallery_empty_bank(self):
        bank = TemporalFeatureBank(window=30)
        ids, gallery = bank.get_gallery()
        assert ids == []
        assert gallery.shape[0] == 0

    def test_remove_identity(self):
        bank = TemporalFeatureBank(window=30)
        bank.update(0, torch.randn(512))
        bank.update(1, torch.randn(512))
        bank.remove(0)
        ids, _ = bank.get_gallery()
        assert 0 not in ids
        assert 1 in ids


# ---------------------------------------------------------------------------
# ReIDPipeline integration tests
# ---------------------------------------------------------------------------

class TestReIDPipeline:
    def test_returns_correct_length(self, pipeline: ReIDPipeline, dummy_crop: Image.Image):
        ids = pipeline.process(
            [dummy_crop, dummy_crop], local_track_ids=[1, 2], cam_id=0
        )
        assert len(ids) == 2

    def test_stable_id_across_frames(self, pipeline: ReIDPipeline, dummy_crop: Image.Image):
        """Same (cam, local_id) must return the same global ID each frame."""
        ids1 = pipeline.process([dummy_crop], local_track_ids=[99], cam_id=0)
        ids2 = pipeline.process([dummy_crop], local_track_ids=[99], cam_id=0)
        assert ids1[0] == ids2[0], "Same local track should map to same global ID"

    def test_mismatched_lengths_raise(self, pipeline: ReIDPipeline, dummy_crop: Image.Image):
        with pytest.raises(ValueError):
            pipeline.process([dummy_crop], local_track_ids=[1, 2], cam_id=0)

    def test_empty_input_returns_empty(self, pipeline: ReIDPipeline):
        ids = pipeline.process([], local_track_ids=[], cam_id=0)
        assert ids == []

    def test_unique_ids_for_new_tracks(
        self,
        pipeline: ReIDPipeline,
        dummy_crop: Image.Image,
        dummy_crop_alt: Image.Image,
    ):
        """Two clearly distinct new detections should not share a global ID."""
        # Use a fresh pipeline with a high threshold so orthogonal embeddings
        # are never matched.
        fresh = ReIDPipeline(
            encoder=pipeline._encoder,
            projector=IdentityProjector(),
            matcher=CosineMatcher(threshold=0.99),
            bank=TemporalFeatureBank(window=30),
        )
        ids = fresh.process(
            [dummy_crop, dummy_crop_alt], local_track_ids=[10, 20], cam_id=0
        )
        assert ids[0] != ids[1], "Different crops should get different global IDs"

    def test_cross_camera_reid(
        self, encoder: CLIPEncoder, dummy_crop: Image.Image
    ):
        """Same appearance crop from cam 0 and cam 1 should converge to the same ID.

        The second camera sees the same person after a gallery entry has been
        built by cam 0, so the cosine matcher should match to the same global ID.
        """
        fresh = ReIDPipeline(
            encoder=encoder,
            projector=IdentityProjector(),
            matcher=CosineMatcher(threshold=0.70),
            bank=TemporalFeatureBank(window=30),
        )
        # Cam 0 sees local track 1 — builds gallery
        ids_cam0 = fresh.process([dummy_crop], local_track_ids=[1], cam_id=0)
        # Cam 1 sees local track 1 — should match against gallery
        ids_cam1 = fresh.process([dummy_crop], local_track_ids=[1], cam_id=1)

        assert ids_cam0[0] == ids_cam1[0], (
            f"Cross-camera re-ID failed: cam0 id={ids_cam0[0]}, cam1 id={ids_cam1[0]}"
        )
