"""Cosine similarity matching for Re-ID.

SRP: responsible solely for comparing query embeddings against a gallery
and assigning best-match identities above a configurable threshold.

Formula:  c_f = (F1 · F2) / (|F1| · |F2|)
"""

from __future__ import annotations

import torch


class CosineMatcher:
    """Matches query features to a gallery via cosine similarity.

    c_f = (F1 · F2) / (|F1| · |F2|)

    Queries whose best match score is below ``threshold`` are unmatched and
    should be assigned a new identity by the caller.

    Args:
        threshold: Minimum cosine similarity required to declare a match.
    """

    UNMATCHED = -1  # sentinel returned when no gallery match exceeds threshold

    def __init__(self, threshold: float = 0.7) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def similarity_matrix(
        self,
        query_feats: torch.Tensor,    # (Q, D)  L2-normalised
        gallery_feats: torch.Tensor,  # (G, D)  L2-normalised
    ) -> torch.Tensor:
        """Return (Q, G) cosine similarity matrix."""
        # Both inputs are expected to be L2-normalised; dot product == cosine sim.
        return query_feats @ gallery_feats.T  # (Q, G)

    def match(
        self,
        query_feats: torch.Tensor,    # (Q, D)  L2-normalised
        gallery_feats: torch.Tensor,  # (G, D)  L2-normalised
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find best-matching gallery index for each query.

        Args:
            query_feats:   (Q, D) tensor of query embeddings.
            gallery_feats: (G, D) tensor of gallery embeddings.

        Returns:
            sim_matrix:  (Q, G) cosine similarity matrix (float).
            assignments: (Q,) index tensor; entry is gallery index [0, G) when
                         the best match exceeds ``threshold``, else UNMATCHED.
        """
        if gallery_feats.shape[0] == 0:
            sim = torch.zeros(query_feats.shape[0], 0, device=query_feats.device)
            assignments = torch.full(
                (query_feats.shape[0],), self.UNMATCHED, dtype=torch.long
            )
            return sim, assignments

        sim = self.similarity_matrix(query_feats, gallery_feats)  # (Q, G)
        best_scores, best_indices = sim.max(dim=1)                # (Q,), (Q,)

        assignments = torch.where(
            best_scores >= self.threshold,
            best_indices,
            torch.tensor(self.UNMATCHED, dtype=torch.long, device=sim.device),
        )
        return sim, assignments
