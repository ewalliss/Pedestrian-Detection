"""PedestrianReIDConfig — single source of truth for the full pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PedestrianReIDConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    market1501_root: Path = Path("/Users/dangnguyen/Downloads/pedestrian/Market-1501-v15.09.15")
    mot_crops_root: Path = Path("data/mot_crops")

    # ── Model ─────────────────────────────────────────────────────────────────
    clip_model_name: str = "openai/clip-vit-base-patch16"
    pretrain_path: Path | None = None          # local .pt override; None = HF hub
    output_dir: Path = Path("output/pedestrian")

    # ── Prompt — GATE: must contain 'person', must NOT contain 'vehicle' ──────
    prompt_template: str = "a photo of a X X X X person"
    n_ctx: int = 4                             # number of learnable context tokens

    # ── SIE ───────────────────────────────────────────────────────────────────
    num_cams: int = 30                         # Market-1501 (6) + MOT cams (10-23)
    num_views: int = 4                         # viewpoint bins
    sie_coe: float = 3.1                       # SIE scale factor (original paper: 3.1)

    # ── OLP ───────────────────────────────────────────────────────────────────
    olp_k: int = 16                            # top-k patches to pool

    # ── Training — Stage 1 ───────────────────────────────────────────────────
    epochs_stage1: int = 120
    lr_stage1: float = 3.5e-4
    warmup_epochs: int = 5
    weight_decay: float = 1e-4

    # ── Training — Stage 2 ───────────────────────────────────────────────────
    epochs_stage2: int = 100
    lr_stage2: float = 5e-6
    lr_decay_steps: list[int] = field(default_factory=lambda: [30, 50])
    lr_decay_gamma: float = 0.1
    early_stop_patience: int = 10

    # ── Shared training ───────────────────────────────────────────────────────
    batch_size: int = 64
    num_instances: int = 4                     # K images per identity (P×K sampler)
    fp16: bool = True
    num_workers: int = 2                       # Windows: keep ≤ 2
    checkpoint_period: int = 10               # save every N epochs

    # ── Loss weights — Stage 2 ────────────────────────────────────────────────
    id_loss_weight: float = 1.0
    circle_loss_weight: float = 1.0
    i2t_loss_weight: float = 1.0
    label_smoothing: float = 0.1

    # ── Circle Loss (Sun et al., CVPR 2020) ──────────────────────────────────
    circle_gamma: int = 256                    # scale factor γ (256 for fine-grained)
    circle_margin: float = 0.25                # relaxation margin m

    # ── I2T contrastive ──────────────────────────────────────────────────────
    i2t_temperature: float = 0.07              # CLIP-style logit temperature

    # ── LR schedule — Stage 2 ────────────────────────────────────────────────
    lr_warmup_epochs_stage2: int = 10            # linear warmup epochs
    lr_min_stage2: float = 1e-7                # cosine decay floor

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_epochs: int = 30
    lora_lr: float = 1e-4
    lora_accum_steps: int = 2

    # ── Matching ──────────────────────────────────────────────────────────────
    match_threshold: float = 0.7              # cosine similarity min for CosineMatcher

    # ── Re-ranking ────────────────────────────────────────────────────────────
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.7              # higher λ → more Jaccard weight (better at ≥85% mAP)

    # ── Post-processing (eval-time) ──────────────────────────────────────────
    flip_tta: bool = True                   # average orig + hflip features
    power_norm_alpha: float = 0.5           # element-wise |x|^α before L2-norm (0 = disabled)
    qe_k: int = 5                           # alpha-QE: top-K gallery for expansion (0 = disabled)
    qe_alpha: float = 3.0                   # alpha-QE: weight exponent

    # ── Mini mode — 1000 images for pipeline smoke test ───────────────────────
    mini: bool = False
    mini_num_ids: int = 100                    # identities to keep in mini mode
    mini_epochs_stage1: int = 3
    mini_epochs_stage2: int = 2

    def __post_init__(self) -> None:
        # GATE 1 & 2: pedestrian-only guard
        if "vehicle" in self.prompt_template.lower():
            raise ValueError("prompt_template must not contain 'vehicle'")
        if "person" not in self.prompt_template.lower():
            raise ValueError("prompt_template must contain 'person'")
        self.output_dir = Path(self.output_dir)
        self.market1501_root = Path(self.market1501_root)
        self.mot_crops_root = Path(self.mot_crops_root)

    @property
    def effective_epochs_stage1(self) -> int:
        return self.mini_epochs_stage1 if self.mini else self.epochs_stage1

    @property
    def effective_epochs_stage2(self) -> int:
        return self.mini_epochs_stage2 if self.mini else self.epochs_stage2
