"""CLIP ViT-B/32 encoder with frozen first transformer layer.

SRP: responsible solely for converting image crops to L2-normalised embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor


class CLIPEncoder:
    """Encodes person crops to 512-dim L2-normalised CLIP ViT-B/32 vectors.

    Uses ``CLIPModel`` (not ``CLIPVisionModel``) so that the output passes
    through CLIP's visual projection head, yielding the true 512-dim
    joint image-text embedding space instead of the raw 768-dim CLS token.

    The first transformer block (``vision_model.encoder.layers[0]``) is frozen
    immediately after loading to provide stable lower-level features during
    domain adaptation and weight-study experiments on human detection.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device ('cpu', 'cuda', or 'mps').
    """

    _EMBED_DIM = 512  # post-projection dimension for ViT-B/32

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = self._detect_device()

        self.device = torch.device(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._freeze_first_layer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._EMBED_DIM

    def encode(self, crops: list[Image.Image]) -> torch.Tensor:
        """Encode a batch of PIL crops to L2-normalised embeddings.

        Args:
            crops: List of PIL Images (person crops, any size).

        Returns:
            Tensor of shape ``(N, 512)`` on self.device, L2-normalised.
        """
        if not crops:
            return torch.zeros(0, self._EMBED_DIM, device=self.device)

        inputs = self.processor(images=crops, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # pooler_output: (N, 768) → visual_projection: (N, 512)
            vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            image_embeds = self.model.visual_projection(vision_out.pooler_output)

        return F.normalize(image_embeds, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_first_layer(self) -> None:
        """Freeze all parameters in the first ViT transformer block (layer 0)."""
        first_layer = self.model.vision_model.encoder.layers[0]
        for param in first_layer.parameters():
            param.requires_grad = False

    @staticmethod
    def _detect_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
