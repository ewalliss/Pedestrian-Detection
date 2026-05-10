"""RandomIdentitySampler — P identities × K images per batch."""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """Yields batches of (P * K) indices with P identities × K images each.

    Args:
        data_source: Dataset with a `pids` attribute (list of int, same length as dataset).
        batch_size:  Total samples per batch (must equal num_instances * num_pids_per_batch).
        num_instances: K — images per identity per batch.
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        num_instances: int,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        # pid → list of sample indices
        self._pid_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, pid in enumerate(data_source.pids):
            self._pid_to_indices[pid].append(idx)

        self._pids = list(self._pid_to_indices.keys())

    def __len__(self) -> int:
        return len(self._pids) * self.num_instances

    def __iter__(self):
        avail = {pid: list(idxs) for pid, idxs in self._pid_to_indices.items()}
        for idxs in avail.values():
            random.shuffle(idxs)

        batch: list[int] = []
        pids = list(self._pids)
        random.shuffle(pids)

        for pid in pids:
            idxs = avail[pid]
            # If fewer than K samples, oversample with replacement
            if len(idxs) < self.num_instances:
                idxs = random.choices(idxs, k=self.num_instances)
            else:
                idxs = idxs[: self.num_instances]
            batch.extend(idxs)
            if len(batch) >= self.batch_size:
                yield from batch
                batch = []

        if len(batch) >= self.num_instances:
            yield from batch

    def state_dict(self) -> dict:
        return {"pids": list(self._pids)}

    def load_state_dict(self, state: dict) -> None:
        self._pids = state["pids"]
