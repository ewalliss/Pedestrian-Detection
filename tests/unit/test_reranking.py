"""Unit tests for k-reciprocal re-ranking."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.reranking import k_reciprocal_rerank


@pytest.fixture
def feats():
    torch.manual_seed(42)
    Q, G, D = 5, 20, 128
    q = torch.randn(Q, D)
    g = torch.randn(G, D)
    # L2 normalise
    q = q / q.norm(dim=1, keepdim=True)
    g = g / g.norm(dim=1, keepdim=True)
    return q, g


def test_output_shape(feats):
    q, g = feats
    dist = k_reciprocal_rerank(q, g, k1=5, k2=3)
    assert dist.shape == (q.shape[0], g.shape[0])


def test_distances_non_negative(feats):
    q, g = feats
    dist = k_reciprocal_rerank(q, g, k1=5, k2=3)
    assert (dist >= 0).all(), "All distances must be non-negative"


def test_identical_query_gallery_distance_near_zero():
    """If query == gallery[0], the reranked distance to gallery[0] should be near 0."""
    D = 64
    feats = torch.randn(10, D)
    feats = feats / feats.norm(dim=1, keepdim=True)

    q = feats[:1]   # query is feats[0]
    g = feats        # gallery contains the same feats

    dist = k_reciprocal_rerank(q, g, k1=5, k2=3)
    # distance from q to itself (gallery[0]) should be smallest
    assert dist[0, 0] == dist[0].min(), "Exact match should have minimum distance"


def test_no_nan_in_output(feats):
    q, g = feats
    dist = k_reciprocal_rerank(q, g, k1=5, k2=3)
    assert not torch.isnan(dist).any(), "Output must not contain NaN"


def test_no_inf_in_output(feats):
    q, g = feats
    dist = k_reciprocal_rerank(q, g, k1=5, k2=3)
    assert not torch.isinf(dist).any(), "Output must not contain Inf"


def test_lambda_zero_equals_original_distance(feats):
    """With lambda_=0, re-ranked dist should equal the original cosine distance."""
    q, g = feats
    dist_rr = k_reciprocal_rerank(q, g, k1=5, k2=3, lambda_=0.0)
    dist_orig = 1.0 - torch.matmul(q, g.T)
    assert torch.allclose(dist_rr, dist_orig, atol=1e-4), (
        "lambda_=0 should yield original cosine distance"
    )


def test_cpu_device(feats):
    q, g = feats
    assert q.device.type == "cpu"
    dist = k_reciprocal_rerank(q, g, k1=5, k2=3)
    assert dist.device.type == "cpu", "Output device should match input device"
