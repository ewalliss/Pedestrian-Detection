"""OLPHead — Overlapping Local Patch feature extractor for occluded pedestrians.

Extracts top-k patch tokens by L2 norm from ViT's last hidden state,
max-pools them into a local feature, then fuses with the global [CLS] feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OLPHead(nn.Module):
    """Top-k overlapping patch pooling + global [CLS] fusion.

    Args:
        patch_dim: Hidden dimension of ViT patch tokens (768 for ViT-B/16).
        out_dim:   Output feature dimension (512 for CLIP projection space).
        k:         Number of top patches to select.
    """

    def __init__(self, patch_dim: int = 768, out_dim: int = 512, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.last_selected_k: int = 0  # exposed for health check

        # Project patch features to out_dim
        self.patch_proj = nn.Linear(patch_dim, out_dim, bias=False)
        # Fuse global (out_dim) + local (out_dim) → out_dim
        self.fusion = nn.Linear(out_dim * 2, out_dim, bias=False)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.xavier_uniform_(self.fusion.weight)

    def forward(
        self,
        cls_feat: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse global CLS feature with local patch features.

        Args:
            cls_feat:     Global feature from visual projection, shape (N, out_dim).
            patch_tokens: Raw patch hidden states from ViT last layer, shape (N, P, patch_dim).
                          P = 196 for 224×224 ViT-B/16.

        Returns:
            Fused feature tensor, shape (N, out_dim).
        """
        N, P, D = patch_tokens.shape
        k = min(self.k, P)

        # Select top-k patches by L2 norm (in FP32 for stability)
        norms = patch_tokens.float().norm(dim=-1)  # (N, P)
        topk_idx = norms.topk(k, dim=-1).indices   # (N, k)

        # Gather top-k patch tokens
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (N, k, D)
        selected = torch.gather(patch_tokens, 1, idx_expanded)   # (N, k, D)

        # Project to out_dim and max-pool over k patches
        local_feat = self.patch_proj(selected.to(cls_feat.dtype))  # (N, k, out_dim)
        local_feat = local_feat.max(dim=1).values                  # (N, out_dim)

        self.last_selected_k = k

        # Fuse global + local
        fused = self.fusion(torch.cat([cls_feat, local_feat], dim=-1))  # (N, out_dim)
        return F.normalize(fused, p=2, dim=-1)
