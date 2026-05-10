"""Public API surface for src.reid."""

from .clip_encoder import CLIPEncoder
from .cosine_matcher import CosineMatcher
from .pipeline import ReIDPipeline
from .projector import IdentityProjector
from .temporal_bank import TemporalFeatureBank

__all__ = [
    "CLIPEncoder",
    "CosineMatcher",
    "IdentityProjector",
    "ReIDPipeline",
    "TemporalFeatureBank",
]
