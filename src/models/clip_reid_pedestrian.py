"""CLIPReIDPedestrianModel — ViT-B/16 + SIE + OLP for pedestrian re-ID.

Two forward modes controlled by stage flag:
  stage=1 → returns (img_feat, text_feat) for SupConLoss
  stage=2 → returns dual cls_scores + multi-stream features for ID + Circle + I2T

BNNeck pattern (Luo et al. "Bag of Tricks"):
  - Metric loss (Circle): receives L2-normalised pre-BN features
  - ID loss: receives post-BN features → classifier

Eval returns concat(bn(global_512), bn(fused_512)) = 1024-dim for retrieval.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from .olp_head import OLPHead
from .prompt_learner import PromptLearner
from .sie_layer import SIELayer


class TextEncoder(nn.Module):
    """Wraps CLIP's text transformer to accept pre-built prompt embeddings."""

    def __init__(self, clip_model: CLIPModel) -> None:
        super().__init__()
        self.transformer = clip_model.text_model.encoder
        self.final_ln = clip_model.text_model.final_layer_norm
        self.text_proj = clip_model.text_projection
        self.pos_embedding = clip_model.text_model.embeddings.position_embedding

    def forward(self, prompt_emb: torch.Tensor) -> torch.Tensor:
        """Encode prompt embeddings to L2-normalised text features.

        Args:
            prompt_emb: shape (N, seq_len, embed_dim)

        Returns:
            Text features, shape (N, 512), L2-normalised.
        """
        seq_len = prompt_emb.shape[1]
        pos_ids = torch.arange(seq_len, device=prompt_emb.device).unsqueeze(0)
        hidden = prompt_emb + self.pos_embedding(pos_ids)

        # Causal mask: each token attends to previous tokens only
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=hidden.device
        ).triu(diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        out = self.transformer(
            inputs_embeds=hidden,
            attention_mask=None,
            causal_attention_mask=causal_mask,
        ).last_hidden_state  # (N, seq_len, D)

        # Extract EOT (last non-padding) position = index -1
        eot_feat = out[:, -1, :]                     # (N, D_text)
        eot_feat = self.final_ln(eot_feat)
        feat = self.text_proj(eot_feat)              # (N, 512)
        return F.normalize(feat, p=2, dim=-1)


class CLIPReIDPedestrianModel(nn.Module):
    """CLIP ViT-B/16 + SIE + OLP pedestrian re-identification model.

    Args:
        num_pids:    Number of training identities.
        num_cams:    Number of distinct camera IDs.
        num_views:   Number of viewpoint bins.
        olp_k:       Top-k patches for OLP head.
        clip_name:   HuggingFace model id.
        template:    Prompt template (must contain 'person').
        n_ctx:       Learnable context tokens per identity.
    """

    EMBED_DIM = 512      # CLIP projection output
    PATCH_DIM = 768      # ViT-B/16 hidden dimension

    def __init__(
        self,
        num_pids: int,
        num_cams: int = 30,
        num_views: int = 4,
        olp_k: int = 16,
        sie_coe: float = 3.1,
        clip_name: str = "openai/clip-vit-base-patch16",
        template: str = "a photo of a X X X X person",
        n_ctx: int = 4,
    ) -> None:
        super().__init__()

        clip = CLIPModel.from_pretrained(clip_name)

        # Image encoder — full ViT-B/16 vision model
        self.image_encoder = clip.vision_model
        self.visual_proj = clip.visual_projection  # 768 → 512

        # Text path
        self.prompt_learner = PromptLearner(num_pids, clip, template, n_ctx)
        self.text_encoder = TextEncoder(clip)

        # SIE — injected after visual projection (scaled by sie_coe per original paper)
        self.sie = SIELayer(num_cams=num_cams, num_views=num_views, embed_dim=self.EMBED_DIM, sie_coe=sie_coe)

        # OLP — fuses [CLS] + patch features
        self.olp = OLPHead(patch_dim=self.PATCH_DIM, out_dim=self.EMBED_DIM, k=olp_k)

        # Global head: operates on 512-dim projected+SIE feature
        self.bn_global = nn.BatchNorm1d(self.EMBED_DIM)
        self.bn_global.bias.requires_grad_(False)
        self.classifier_global = nn.Linear(self.EMBED_DIM, num_pids, bias=False)
        nn.init.normal_(self.classifier_global.weight, std=0.001)

        # Projected head: operates on 512-dim OLP-fused feature
        self.bn_proj = nn.BatchNorm1d(self.EMBED_DIM)
        self.bn_proj.bias.requires_grad_(False)
        self.classifier_proj = nn.Linear(self.EMBED_DIM, num_pids, bias=False)
        nn.init.normal_(self.classifier_proj.weight, std=0.001)

        self.num_pids = num_pids

    # ── Stage control ─────────────────────────────────────────────────────────

    def freeze_for_stage1(self) -> None:
        """Stage 1: freeze encoders, train only PromptLearner."""
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.visual_proj.parameters():
            p.requires_grad = False
        for p in self.sie.parameters():
            p.requires_grad = False
        for p in self.olp.parameters():
            p.requires_grad = False
        for p in self.bn_global.parameters():
            p.requires_grad = False
        for p in self.classifier_global.parameters():
            p.requires_grad = False
        for p in self.bn_proj.parameters():
            p.requires_grad = False
        for p in self.classifier_proj.parameters():
            p.requires_grad = False
        # PromptLearner remains trainable
        for p in self.prompt_learner.parameters():
            p.requires_grad = True

    def freeze_for_stage2(self) -> None:
        """Stage 2: freeze text path, train image encoder + SIE + OLP + heads."""
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.prompt_learner.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = True
        for p in self.visual_proj.parameters():
            p.requires_grad = True
        for p in self.sie.parameters():
            p.requires_grad = True
        for p in self.olp.parameters():
            p.requires_grad = True
        for p in self.bn_global.parameters():
            p.requires_grad = True
        for p in self.classifier_global.parameters():
            p.requires_grad = True
        for p in self.bn_proj.parameters():
            p.requires_grad = True
        for p in self.classifier_proj.parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract normalised image features via ViT + SIE + OLP.

        Returns:
            (global_feat, fused_feat):
                global_feat: (N, 512) projected + SIE, L2-normalised.
                fused_feat:  (N, 512) OLP-fused, L2-normalised.
        """
        vision_out = self.image_encoder(pixel_values=pixel_values, output_hidden_states=False)
        cls_raw = vision_out.pooler_output          # (N, 768)
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # (N, 196, 768)

        # Project CLS to 512-d + SIE conditioning (single L2-norm AFTER SIE)
        global_feat = self.visual_proj(cls_raw)     # (N, 512)
        global_feat = self.sie(global_feat, cam_ids, view_ids)
        global_feat = F.normalize(global_feat, p=2, dim=-1)

        # OLP: fuse global CLS + top-k patch features
        fused_feat = self.olp(global_feat, patch_tokens)  # (N, 512) L2-normed inside OLP

        return global_feat, fused_feat

    def encode_text(self, pids: torch.Tensor | None = None) -> torch.Tensor:
        """Encode learnable prompts for given pids.

        Returns:
            Text features, shape (N_pids, 512), L2-normalised.
        """
        prompt_emb = self.prompt_learner(pids)     # (N, seq_len, D)
        return self.text_encoder(prompt_emb)       # (N, 512)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pids: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with stage-aware routing.

        Stage 1 (prompt_learner.requires_grad=True, image_encoder frozen):
            Returns img_feat, text_feat, batch_pids for SupConLoss.

        Stage 2 (image_encoder unfrozen):
            Returns dual cls_scores + multi-stream features for ID + Circle + I2T.
            BNNeck: Circle Loss receives pre-BN L2-normed features,
                    ID loss receives post-BN features via classifiers.
        """
        global_feat, fused_feat = self.encode_image(pixel_values, cam_ids, view_ids)

        # Stage 1: return image + text features for contrastive alignment
        if self.prompt_learner.class_ctx.requires_grad and not self.image_encoder.encoder.layers[0].self_attn.k_proj.weight.requires_grad:
            unique_pids = pids.unique(sorted=True)
            text_feat = self.encode_text(unique_pids)
            return {
                "img_feat": fused_feat,
                "text_feat": text_feat,
                "batch_pids": unique_pids,
            }

        # Stage 2: dual-head classification + metric features
        cls_score = self.classifier_global(self.bn_global(global_feat))
        cls_score_proj = self.classifier_proj(self.bn_proj(fused_feat))

        return {
            "global_feat": global_feat,
            "fused_feat": fused_feat,
            "cls_score": cls_score,
            "cls_score_proj": cls_score_proj,
        }

    @torch.no_grad()
    def extract_features(
        self,
        pixel_values: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-only: extract concat(bn(global_512), bn(fused_512)) = 1024-dim features."""
        global_feat, fused_feat = self.encode_image(pixel_values, cam_ids, view_ids)
        feat_bn_global = self.bn_global(global_feat)     # (N, 512)
        feat_bn_proj = self.bn_proj(fused_feat)          # (N, 512)
        return torch.cat([feat_bn_global, feat_bn_proj], dim=1)  # (N, 1024)
