"""PromptLearner — per-identity learnable text tokens for CLIP-ReID.

Template: "a photo of a X X X X person"
The X tokens are learned per-batch (shared context) while the class-specific
tokens are learned per identity.

GATE: raises ValueError if template contains 'vehicle'.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTokenizer


class PromptLearner(nn.Module):
    """Learnable prompt embeddings for pedestrian re-identification.

    Args:
        num_pids:    Number of training identities.
        clip_model:  HuggingFace CLIPModel (used to extract embedding layer).
        template:    Prompt template; X marks learnable token positions.
        n_ctx:       Number of learnable context tokens (replaces X positions).
    """

    def __init__(
        self,
        num_pids: int,
        clip_model: nn.Module,
        template: str = "a photo of a X X X X person",
        n_ctx: int = 4,
    ) -> None:
        super().__init__()

        if "vehicle" in template.lower():
            raise ValueError("PromptLearner: template must not contain 'vehicle'")
        if "person" not in template.lower():
            raise ValueError("PromptLearner: template must contain 'person'")

        self.n_ctx = n_ctx
        self.num_pids = num_pids

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        embed_layer = clip_model.text_model.embeddings.token_embedding
        embed_dim = embed_layer.embedding_dim  # 512 for ViT-B/16

        # Build prefix / suffix from template (strip X placeholders)
        prefix_text = template.split("X")[0].strip()   # "a photo of a"
        suffix_text = template.split("X")[-1].strip()  # "person"

        with torch.no_grad():
            prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            sos_id = torch.tensor([[tokenizer.bos_token_id]])
            eos_id = torch.tensor([[tokenizer.eos_token_id]])

            prefix_emb = embed_layer(prefix_ids)   # (1, L_pre, D)
            suffix_emb = embed_layer(suffix_ids)   # (1, L_suf, D)
            sos_emb = embed_layer(sos_id)           # (1, 1, D)
            eos_emb = embed_layer(eos_id)           # (1, 1, D)

        self.register_buffer("prefix_emb", prefix_emb.squeeze(0))  # (L_pre, D)
        self.register_buffer("suffix_emb", suffix_emb.squeeze(0))  # (L_suf, D)
        self.register_buffer("sos_emb", sos_emb.squeeze(0))        # (1, D)
        self.register_buffer("eos_emb", eos_emb.squeeze(0))        # (1, D)

        # Learnable: shared context + per-class context
        self.ctx = nn.Parameter(torch.empty(n_ctx, embed_dim).normal_(std=0.02))
        self.class_ctx = nn.Parameter(torch.empty(num_pids, n_ctx, embed_dim).normal_(std=0.02))

        # Compute sequence length for position IDs
        self._seq_len = 1 + prefix_emb.shape[1] + n_ctx + suffix_emb.shape[1] + 1  # SOS+pre+ctx+suf+EOS
        self.register_buffer(
            "position_ids",
            torch.arange(self._seq_len).unsqueeze(0),  # (1, seq_len)
        )

    def forward(self, pids: torch.Tensor | None = None) -> torch.Tensor:
        """Build prompt embeddings for given pids (or all pids if None).

        Returns:
            Tensor of shape (N_pids, seq_len, embed_dim)
        """
        if pids is None:
            class_ctx = self.class_ctx                     # (num_pids, n_ctx, D)
        else:
            class_ctx = self.class_ctx[pids]               # (B, n_ctx, D)

        N = class_ctx.shape[0]
        ctx = self.ctx.unsqueeze(0).expand(N, -1, -1)     # (N, n_ctx, D)
        combined_ctx = ctx + class_ctx                    # (N, n_ctx, D)

        sos = self.sos_emb.unsqueeze(0).expand(N, -1, -1)    # (N, 1, D)
        pre = self.prefix_emb.unsqueeze(0).expand(N, -1, -1) # (N, L_pre, D)
        suf = self.suffix_emb.unsqueeze(0).expand(N, -1, -1) # (N, L_suf, D)
        eos = self.eos_emb.unsqueeze(0).expand(N, -1, -1)    # (N, 1, D)

        prompt_emb = torch.cat([sos, pre, combined_ctx, suf, eos], dim=1)  # (N, seq_len, D)
        return prompt_emb

    @property
    def seq_len(self) -> int:
        return self._seq_len
