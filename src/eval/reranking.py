"""K-reciprocal re-ranking (Zhong et al. CVPR 2017).

All computation runs on CPU FP32 to avoid GPU OOM on large galleries.
k1=20, k2=6, lambda=0.3 as recommended in the original paper.
"""

from __future__ import annotations

import torch


def k_reciprocal_rerank(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_: float = 0.3,
) -> torch.Tensor:
    """Re-rank gallery for each query using k-reciprocal encoding.

    Args:
        query_feats:   (Q, D) L2-normalised query features.
        gallery_feats: (G, D) L2-normalised gallery features.
        k1:            K-reciprocal neighbourhood size.
        k2:            Local query expansion size.
        lambda_:       Weight for Jaccard distance.

    Returns:
        Re-ranked distance matrix, shape (Q, G), lower = more similar.
    """
    query_feats = query_feats.cpu().float()
    gallery_feats = gallery_feats.cpu().float()
    device = query_feats.device
    Q = query_feats.shape[0]
    G = gallery_feats.shape[0]

    # Stack all features for joint neighbour computation
    all_feats = torch.cat([query_feats, gallery_feats], dim=0)  # (Q+G, D)
    N = all_feats.shape[0]

    # Cosine distance (CPU FP32)
    sim = torch.matmul(all_feats, all_feats.T)   # (N, N)
    orig_dist = (1.0 - sim).clamp(min=0.0)       # (N, N)
    del sim

    # K-reciprocal sets: V[i] ∈ R^N (Gaussian-weighted)
    V = torch.zeros(N, N, device=device)

    # Sorted neighbours (ascending distance)
    _, nn_idx = orig_dist.topk(N, dim=1, largest=False)  # (N, N)

    for i in range(N):
        # R(i, k1): k1-nearest neighbours of i
        fwd_k1 = set(nn_idx[i, 1 : k1 + 1].tolist())

        # k-reciprocal neighbours: j in fwd_k1 AND i in R(j, k1)
        recip = set()
        for j in fwd_k1:
            back_k1 = set(nn_idx[j, 1 : k1 + 1].tolist())
            if i in back_k1:
                recip.add(j)

        # Local query expansion: merge k2-NN of reciprocal neighbours
        expanded = set(recip)
        for j in list(recip):
            back_k2 = set(nn_idx[j, 1 : k2 + 1].tolist())
            overlap = back_k2 & recip
            if len(overlap) >= 2.0 / 3.0 * len(recip):
                expanded |= back_k2
        recip_final = sorted(expanded)

        if recip_final:
            # Gaussian kernel weights based on rank distance
            dists_to_recip = orig_dist[i, recip_final]
            weight = torch.exp(-dists_to_recip)
            V[i, recip_final] = weight / weight.sum().clamp(min=1e-8)

    # Jaccard distance between query and gallery V vectors
    V_q = V[:Q]                  # (Q, N)
    V_g = V[Q:]                  # (G, N)  — not used directly; use all-N V

    # Full V similarity: dot product (both are non-negative)
    # JD(q, g) = 1 - |V_q ∩ V_g| / |V_q ∪ V_g|
    # ≈ 1 - 2 * dot(V_q, V_g) / (sum(V_q) + sum(V_g))
    dot = torch.matmul(V[:Q], V[Q:].T)              # (Q, G)
    norm_q = V[:Q].sum(dim=1, keepdim=True)         # (Q, 1)
    norm_g = V[Q:].sum(dim=1, keepdim=True)         # (G, 1)
    jaccard = 1.0 - 2.0 * dot / (norm_q + norm_g.T).clamp(min=1e-8)

    # Original distance (query vs gallery only)
    orig_qg = orig_dist[:Q, Q:]                 # (Q, G)

    return (1.0 - lambda_) * orig_qg + lambda_ * jaccard
