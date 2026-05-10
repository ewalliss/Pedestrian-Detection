"""End-to-end Re-ID pipeline: crop -> encode -> match -> update bank -> global IDs.

SRP: orchestrates CLIPEncoder, IdentityProjector, CosineMatcher, TemporalFeatureBank.
DIP: all components are injected via the constructor — no hardcoded dependencies.
"""

from __future__ import annotations

import itertools

import torch
from PIL import Image

from .clip_encoder import CLIPEncoder
from .cosine_matcher import CosineMatcher
from .projector import IdentityProjector
from .temporal_bank import TemporalFeatureBank


class ReIDPipeline:
    """Assigns globally-unique Re-ID labels to per-camera local track IDs.

    Workflow per ``process()`` call:
    1. Encode crops with CLIP encoder.
    2. Project embeddings (optional linear head).
    3. Build gallery from temporal bank (mean of last 30 frames per identity).
    4. Match queries against gallery via cosine similarity.
    5. Update bank with new embeddings; allocate new IDs for unmatched queries.
    6. Return list of global identity IDs aligned with input crops.

    Cross-camera re-identification is handled by namespacing local track IDs as
    ``(cam_id, local_track_id)`` before gallery lookup, while the global identity
    ID space is shared.

    Args:
        encoder:   CLIPEncoder instance (frozen layer 0).
        projector: IdentityProjector instance (identity by default).
        matcher:   CosineMatcher with configured similarity threshold.
        bank:      TemporalFeatureBank with configured window size.
    """

    def __init__(
        self,
        encoder: CLIPEncoder,
        projector: IdentityProjector,
        matcher: CosineMatcher,
        bank: TemporalFeatureBank,
    ) -> None:
        self._encoder = encoder
        self._projector = projector
        self._matcher = matcher
        self._bank = bank

        # Global identity counter; monotonically increasing.
        self._id_counter = itertools.count()

        # Maps (cam_id, local_track_id) -> global_identity_id
        self._local_to_global: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        crops: list[Image.Image],
        local_track_ids: list[int],
        cam_id: int = 0,
    ) -> list[int]:
        """Assign or look up global identity IDs for a batch of crops.

        Args:
            crops:           Person crop images aligned with ``local_track_ids``.
            local_track_ids: Tracker-assigned IDs for this camera frame.
            cam_id:          Camera index; disambiguates IDs across cameras.

        Returns:
            List of global identity integers, same length as ``crops``.
        """
        if len(crops) != len(local_track_ids):
            raise ValueError(
                f"crops and local_track_ids must have the same length, "
                f"got {len(crops)} vs {len(local_track_ids)}"
            )

        if not crops:
            return []

        # 1. Encode
        raw_feats = self._encoder.encode(crops)           # (N, 512)
        # 2. Project
        feats = self._projector.project(raw_feats)        # (N, D)
        # 3. Gallery
        gallery_ids, gallery_feats = self._bank.get_gallery()  # (K,), (K, D)

        if gallery_feats.shape[0] > 0:
            gallery_feats = gallery_feats.to(feats.device)

        # 4. Match
        _, assignments = self._matcher.match(feats, gallery_feats)  # (N,)

        global_ids: list[int] = []

        for i, (local_id, feat, assignment) in enumerate(
            zip(local_track_ids, feats, assignments)
        ):
            key = (cam_id, local_id)

            if key in self._local_to_global:
                # Known track from this camera: keep stable global ID
                gid = self._local_to_global[key]
            elif assignment.item() != CosineMatcher.UNMATCHED:
                # Matched to an existing identity in the bank
                gid = gallery_ids[assignment.item()]
                self._local_to_global[key] = gid
            else:
                # New identity
                gid = next(self._id_counter)
                self._local_to_global[key] = gid

            # 5. Update bank
            self._bank.update(gid, feat.squeeze(0))
            global_ids.append(gid)

        return global_ids
