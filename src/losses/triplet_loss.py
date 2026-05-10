"""Triplet loss with online hard-mining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Batch-hard triplet loss with cosine distance.

    For each anchor, selects the hardest positive (most dissimilar same-id)
    and hardest negative (most similar different-id) within the batch.

    Args:
        margin: Triplet margin.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch-hard triplet loss.

        Args:
            feats:  (N, D) L2-normalised feature vectors.
            labels: (N,) identity labels.

        Returns:
            Scalar loss.
        """
        # Pairwise cosine distance = 1 - sim (feats already L2-normalised)
        dist = 1.0 - torch.matmul(feats, feats.T)  # (N, N)

        same_id = labels.unsqueeze(1) == labels.unsqueeze(0)   # (N, N)
        diff_id = ~same_id

        # Hard positive: largest distance among same-id pairs (excluding self)
        pos_dist = dist.clone()
        pos_dist[~same_id] = -1.0
        pos_dist.fill_diagonal_(-1.0)
        hard_pos = pos_dist.max(dim=1).values  # (N,)

        # Hard negative: smallest distance among diff-id pairs
        neg_dist = dist.clone()
        neg_dist[same_id] = float("inf")
        hard_neg = neg_dist.min(dim=1).values  # (N,)

        loss = F.relu(hard_pos - hard_neg + self.margin)

        # Only include anchors that have both a valid positive and negative
        valid = (hard_pos > -1.0) & (hard_neg < float("inf"))
        return loss[valid].mean() if valid.any() else loss.new_tensor(0.0)
