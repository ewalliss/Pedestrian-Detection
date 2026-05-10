"""Circle Loss — unified pair similarity optimisation (Sun et al., CVPR 2020).

SRP: one loss class, pair-wise mode only (used for Re-ID metric learning).
DIP: hyperparameters injected via constructor, not hardcoded.

Reference:
    Sun et al. "Circle Loss: A Unified Perspective of Pair Similarity Optimization."
    CVPR 2020. arXiv:2002.10857.
    Recommended Re-ID hyperparams: γ=128, m=0.25.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """Pair-wise Circle Loss for Re-ID metric learning.

    For each anchor the loss simultaneously:
    - Pushes all positive similarities s_p → 1
    - Pulls all negative similarities s_n → 0

    Self-paced weights give larger gradients to the most problematic pairs
    (near-miss negatives, poorly-separated positives), unlike batch-hard
    triplet which treats all active pairs equally.

    Decision boundary: (s_n)² + (s_p − 1)² = 2m²  (a circle in (s_n, s_p) space).

    Formula (pair-wise mode, Eq. 6 in paper):
        L = log[ 1 + Σ_j exp(γ·αₙʲ·(sₙʲ + m)) · Σᵢ exp(−γ·αₚⁱ·(sₚⁱ − (1−m))) ]
        αₚ = [1+m − sₚ]₊      (emphasises under-optimised positives)
        αₙ = [sₙ + m]₊        (emphasises near-miss negatives)

    Implemented via logsumexp + softplus for full numerical stability at γ=128
    (naïve exp would overflow: γ·αₙ·sₙ can reach ~200).

    Args:
        gamma:  Scale factor γ. Paper recommends 128 for Re-ID.
        margin: Relaxation margin m. Paper recommends 0.25 for Re-ID.
    """

    GAMMA_DEFAULT: int = 128
    MARGIN_DEFAULT: float = 0.25

    def __init__(self, gamma: int = GAMMA_DEFAULT, margin: float = MARGIN_DEFAULT) -> None:
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        # Precompute fixed boundary offsets
        self._op = 1.0 + margin   # optimal positive target (1 + m)
        self._on = -margin        # optimal negative target (−m)
        self._dp = 1.0 - margin   # positive decision boundary offset (1 − m)
        self._dn = margin         # negative decision boundary offset (m)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute pair-wise Circle Loss over a mini-batch.

        Inputs must be L2-normalised so all pairwise similarities are in [−1, 1].

        Args:
            feats:  (N, D) L2-normalised feature vectors.
            labels: (N,) integer identity labels.

        Returns:
            Scalar loss averaged over anchors that have at least one positive
            and one negative within the batch.
        """
        # (N, N) cosine similarity matrix (feats already L2-normalised)
        sim = torch.matmul(feats, feats.T)  # (N, N)

        same = labels.unsqueeze(1) == labels.unsqueeze(0)  # (N, N) bool
        pos_mask = same & ~torch.eye(len(labels), dtype=torch.bool, device=feats.device)
        neg_mask = ~same

        # Self-paced weights — clamp negatives to zero
        # α_p = [O_p − s_p]₊ = [1+m − s_p]₊
        alpha_p = F.relu(self._op - sim.detach())   # (N, N)
        # α_n = [s_n − O_n]₊ = [s_n + m]₊
        alpha_n = F.relu(sim.detach() - self._on)   # (N, N)

        # Masked log-sum-exp terms (−inf for masked entries → contributes 0 after logsumexp)
        _neg_inf = float("-inf")

        # Negative term: Σ_j exp(γ · α_n^j · (s_n^j − Δ_n))  where Δ_n = m
        neg_logit = self.gamma * alpha_n * (sim - self._dn)   # (N, N)
        neg_logit = neg_logit.masked_fill(~neg_mask, _neg_inf)
        neg_term = neg_logit.logsumexp(dim=1)                 # (N,)

        # Positive term: Σ_i exp(−γ · α_p^i · (s_p^i − Δ_p))  where Δ_p = 1−m
        pos_logit = -self.gamma * alpha_p * (sim - self._dp)  # (N, N)
        pos_logit = pos_logit.masked_fill(~pos_mask, _neg_inf)
        pos_term = pos_logit.logsumexp(dim=1)                 # (N,)

        # L = log(1 + exp(neg_term + pos_term)) = softplus(neg_term + pos_term)
        # softplus is numerically stable for large arguments
        loss = F.softplus(neg_term + pos_term)                # (N,)

        # Only include anchors with at least one positive AND one negative
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        return loss[valid].mean() if valid.any() else loss.new_tensor(0.0)
