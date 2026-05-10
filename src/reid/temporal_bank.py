"""Rolling temporal feature bank per identity.

SRP: responsible solely for storing and retrieving the smoothed temporal
feature representation of tracked identities.

Each identity is represented by the mean of its last ``window`` CLIP embeddings,
implementing the temporal cosine similarity:
    c_f = (F1 · F2) / (|F1| · |F2|)
where F2 is the mean gallery vector from this bank.
"""

from __future__ import annotations

from collections import defaultdict, deque

import torch
import torch.nn.functional as F


class TemporalFeatureBank:
    """Rolling window of per-identity CLIP feature vectors.

    Args:
        window: Maximum number of frames to retain per identity (default: 30,
                ~1 second at 30 fps). Older frames are evicted automatically.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._bank: dict[int, deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, identity_id: int, feature: torch.Tensor) -> None:
        """Append a new feature vector for ``identity_id``.

        Args:
            identity_id: Integer identity key.
            feature:     1-D tensor of shape ``(D,)``, L2-normalised.
        """
        self._bank[identity_id].append(feature.detach().cpu())

    def get_gallery(self) -> tuple[list[int], torch.Tensor]:
        """Return identity IDs and their mean feature matrix.

        Returns:
            id_list:      List of K identity IDs in insertion order.
            mean_features: Tensor of shape ``(K, D)``, L2-normalised means.
                           Returns empty tensors when the bank is empty.
        """
        if not self._bank:
            return [], torch.zeros(0, 0)

        id_list: list[int] = []
        means: list[torch.Tensor] = []

        for identity_id, frames in self._bank.items():
            stacked = torch.stack(list(frames), dim=0)  # (T, D)
            mean_feat = stacked.mean(dim=0)             # (D,)
            mean_feat = F.normalize(mean_feat, p=2, dim=0)
            id_list.append(identity_id)
            means.append(mean_feat)

        return id_list, torch.stack(means, dim=0)  # (K, D)

    def remove(self, identity_id: int) -> None:
        """Evict an identity from the bank (e.g. after track is lost)."""
        self._bank.pop(identity_id, None)

    def __len__(self) -> int:
        return len(self._bank)
