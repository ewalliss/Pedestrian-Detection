"""SIELayer — Side Information Embedding for camera/viewpoint conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class SIELayer(nn.Module):
    """Additive camera + viewpoint embedding injected into visual features.

    Args:
        num_cams:  Total number of camera IDs (across all datasets).
        num_views: Total number of viewpoint bins.
        embed_dim: Feature dimension to match (512 for CLIP ViT-B/16).
    """

    def __init__(self, num_cams: int, num_views: int, embed_dim: int = 512, sie_coe: float = 3.1) -> None:
        super().__init__()
        self.sie_coe = sie_coe
        self.cam_embedding = nn.Embedding(num_cams, embed_dim)
        self.view_embedding = nn.Embedding(num_views, embed_dim)
        nn.init.normal_(self.cam_embedding.weight, std=0.01)
        nn.init.normal_(self.view_embedding.weight, std=0.01)

    def forward(
        self,
        feat: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add scaled camera and viewpoint embeddings to visual feature.

        Args:
            feat:     Visual feature tensor, shape (N, embed_dim).
            cam_ids:  Camera indices, shape (N,).
            view_ids: Viewpoint indices, shape (N,).

        Returns:
            Conditioned feature, shape (N, embed_dim).
        """
        assert cam_ids.max() < self.cam_embedding.num_embeddings, (
            f"cam_id {cam_ids.max().item()} >= num_cams {self.cam_embedding.num_embeddings}"
        )
        return feat + self.sie_coe * (self.cam_embedding(cam_ids) + self.view_embedding(view_ids))
