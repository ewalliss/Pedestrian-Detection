"""Unit tests for OLPHead (Overlapping Local Patches)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.olp_head import OLPHead


@pytest.fixture
def olp():
    return OLPHead(patch_dim=768, feat_dim=512, k=8)


def test_output_shape(olp):
    B = 16
    cls_feat = torch.randn(B, 512)
    patch_tokens = torch.randn(B, 196, 768)   # ViT-B/16: 196 patches
    out = olp(cls_feat, patch_tokens)
    assert out.shape == (B, 512), f"Expected (16, 512), got {out.shape}"


def test_output_is_l2_normalized(olp):
    cls_feat = torch.randn(8, 512)
    patch_tokens = torch.randn(8, 196, 768)
    out = olp(cls_feat, patch_tokens)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), "Output should be L2-normalised"


def test_last_selected_k_updated(olp):
    olp(torch.randn(4, 512), torch.randn(4, 196, 768))
    assert olp.last_selected_k == 8, f"Expected k=8, got {olp.last_selected_k}"


def test_last_selected_k_clipped_to_num_patches(olp):
    """When k > num_patches, should select all patches."""
    small_olp = OLPHead(patch_dim=768, feat_dim=512, k=1000)
    small_olp(torch.randn(2, 512), torch.randn(2, 5, 768))
    assert small_olp.last_selected_k == 5, "k should be clipped to actual number of patches"


def test_no_nan_in_output(olp):
    cls_feat = torch.randn(8, 512)
    patch_tokens = torch.randn(8, 196, 768)
    out = olp(cls_feat, patch_tokens)
    assert not torch.isnan(out).any(), "Output should not contain NaN"


def test_deterministic(olp):
    cls_feat = torch.randn(4, 512)
    patch_tokens = torch.randn(4, 196, 768)
    out1 = olp(cls_feat, patch_tokens)
    out2 = olp(cls_feat, patch_tokens)
    assert torch.allclose(out1, out2), "OLP should be deterministic"


def test_gradient_flows(olp):
    cls_feat = torch.randn(4, 512, requires_grad=True)
    patch_tokens = torch.randn(4, 196, 768, requires_grad=True)
    out = olp(cls_feat, patch_tokens)
    out.sum().backward()
    assert cls_feat.grad is not None, "Gradient should flow to cls_feat"
    assert patch_tokens.grad is not None, "Gradient should flow to patch_tokens"
