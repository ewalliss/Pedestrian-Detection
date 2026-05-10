"""Linear projection head: maps CLIP embeddings to a target dimension.

SRP: responsible solely for dimensionality projection.

Default is identity (in_dim == out_dim) so the module is a no-op unless
you explicitly pass different dimensions for fine-tuning experiments (OCP).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityProjector:
    """Projects feature vectors from ``in_dim`` to ``out_dim``.

    When ``in_dim == out_dim`` the internal linear layer acts as an identity
    mapping (weights initialised to the identity matrix, bias to zero).

    Args:
        in_dim:  Input dimensionality (512 for ViT-B/32 CLS token).
        out_dim: Output dimensionality. Defaults to ``in_dim`` (no projection).
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 512) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

        self._linear: nn.Linear | None = None
        if in_dim != out_dim:
            self._linear = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.xavier_uniform_(self._linear.weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Project features and L2-normalise the result.

        Args:
            features: Tensor of shape ``(N, in_dim)``.

        Returns:
            Tensor of shape ``(N, out_dim)``, L2-normalised.
        """
        if self._linear is not None:
            features = self._linear(features)
        return F.normalize(features, p=2, dim=-1)
