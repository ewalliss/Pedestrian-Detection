"""Unit tests for SIELayer."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.sie_layer import SIELayer


@pytest.fixture
def sie():
    return SIELayer(num_cams=10, num_views=4, embed_dim=512)


def test_output_shape(sie):
    B = 8
    feat = torch.randn(B, 512)
    cam_ids = torch.randint(0, 10, (B,))
    view_ids = torch.randint(0, 4, (B,))
    out = sie(feat, cam_ids, view_ids)
    assert out.shape == (B, 512), f"Expected (8, 512), got {out.shape}"


def test_different_cams_give_different_output(sie):
    feat = torch.zeros(2, 512)
    cam_a = torch.tensor([0, 0])
    cam_b = torch.tensor([1, 1])
    view = torch.zeros(2, dtype=torch.long)
    out_a = sie(feat, cam_a, view)
    out_b = sie(feat, cam_b, view)
    assert not torch.allclose(out_a, out_b), "Different cameras should produce different outputs"


def test_different_views_give_different_output(sie):
    feat = torch.zeros(2, 512)
    cam = torch.zeros(2, dtype=torch.long)
    view_a = torch.tensor([0, 0])
    view_b = torch.tensor([2, 2])
    out_a = sie(feat, cam, view_a)
    out_b = sie(feat, cam, view_b)
    assert not torch.allclose(out_a, out_b), "Different views should produce different outputs"


def test_same_input_same_output(sie):
    feat = torch.randn(4, 512)
    cam_ids = torch.tensor([0, 1, 2, 3])
    view_ids = torch.zeros(4, dtype=torch.long)
    out1 = sie(feat, cam_ids, view_ids)
    out2 = sie(feat, cam_ids, view_ids)
    assert torch.allclose(out1, out2), "Deterministic: same input → same output"


def test_weights_nonzero_after_init(sie):
    cam_w = sie.cam_embedding.weight
    view_w = sie.view_embedding.weight
    assert cam_w.abs().sum() > 0, "Camera embedding weights should be non-zero after init"
    assert view_w.abs().sum() > 0, "View embedding weights should be non-zero after init"


def test_additive_decomposition(sie):
    """Output should be feat + cam_emb + view_emb."""
    feat = torch.randn(1, 512)
    cam_ids = torch.tensor([3])
    view_ids = torch.tensor([1])
    out = sie(feat, cam_ids, view_ids)
    expected = feat + sie.cam_embedding(cam_ids) + sie.view_embedding(view_ids)
    assert torch.allclose(out, expected), "SIE output should be feat + cam_emb + view_emb"
