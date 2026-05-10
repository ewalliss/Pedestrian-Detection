"""Identity classification loss with label smoothing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDLoss(nn.Module):
    """Cross-entropy with label smoothing for identity classification.

    Args:
        num_classes: Number of training identities.
        epsilon:     Label smoothing factor.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy.

        Args:
            logits: (N, num_classes) raw classification scores.
            labels: (N,) ground-truth identity indices.

        Returns:
            Scalar loss.
        """
        log_probs = self.log_softmax(logits)           # (N, C)

        # One-hot smoothed targets
        targets = torch.zeros_like(log_probs)
        targets.fill_(self.epsilon / (self.num_classes - 1))
        targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.epsilon)

        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss
