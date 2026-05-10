"""Supervised Contrastive Loss for image-text alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss between two feature sets.

    Positive pairs are defined by matching labels.
    Used for both i2t (image-to-text) and t2i (text-to-image) directions.

    Args:
        temperature: Logit scaling factor.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss from A→B.

        Args:
            feats_a:   (N, D) L2-normalised features (anchor).
            feats_b:   (M, D) L2-normalised features (target).
            labels_a:  (N,) identity labels for anchors.
            labels_b:  (M,) identity labels for targets.

        Returns:
            Scalar loss.
        """
        # Similarity matrix (N, M)
        logits = torch.matmul(feats_a, feats_b.T) / self.temperature

        # Positive mask: (N, M), True where labels match
        mask = (labels_a.unsqueeze(1) == labels_b.unsqueeze(0)).float()

        # At least one positive per anchor required
        has_positive = mask.sum(dim=1) > 0
        if not has_positive.any():
            return logits.new_tensor(0.0)

        # Log-sum-exp over all pairs
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Mean log-prob over positives
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return loss[has_positive].mean()
