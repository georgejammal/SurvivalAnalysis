from __future__ import annotations

import numpy as np
import torch


class FeatureMasker:
    def __init__(
        self,
        n_features: int,
        *,
        min_ctx_share: float = 0.6,
        max_ctx_share: float = 0.8,
        min_tgt_share: float = 0.2,
        max_tgt_share: float = 0.4,
        allow_overlap: bool = False,
        feature_sampling_weights: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.n = int(n_features)
        self.min_ctx = int(round(self.n * float(min_ctx_share)))
        self.max_ctx = int(round(self.n * float(max_ctx_share)))
        self.min_tgt = int(round(self.n * float(min_tgt_share)))
        self.max_tgt = int(round(self.n * float(max_tgt_share)))
        self.allow_overlap = bool(allow_overlap)
        self.rng = np.random.default_rng(seed)
        if feature_sampling_weights is not None:
            w = np.asarray(feature_sampling_weights, dtype=float)
            if w.shape != (self.n,):
                raise ValueError("feature_sampling_weights must have shape (n_features,)")
            w = np.clip(w, 1e-12, None)
            self.w = w / w.sum()
        else:
            self.w = None

    def _sample_indices(self, k: int, available: np.ndarray) -> np.ndarray:
        if self.w is None:
            return self.rng.choice(available, size=k, replace=False)
        probs = self.w[available]
        probs = probs / probs.sum()
        return self.rng.choice(available, size=k, replace=False, p=probs)

    def sample_masks(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask_ctx = np.zeros((batch_size, self.n), dtype=np.float32)
        mask_tgt = np.zeros((batch_size, self.n), dtype=np.float32)

        for i in range(batch_size):
            n_ctx = int(self.rng.integers(self.min_ctx, self.max_ctx + 1))
            n_tgt = int(self.rng.integers(self.min_tgt, self.max_tgt + 1))

            all_idx = np.arange(self.n)
            ctx_idx = self._sample_indices(n_ctx, all_idx)
            mask_ctx[i, ctx_idx] = 1.0

            if self.allow_overlap:
                tgt_available = all_idx
            else:
                tgt_available = np.setdiff1d(all_idx, ctx_idx, assume_unique=False)
                if tgt_available.size < n_tgt:
                    # fall back to allow overlap if needed
                    tgt_available = all_idx
            tgt_idx = self._sample_indices(n_tgt, tgt_available)
            mask_tgt[i, tgt_idx] = 1.0

        return torch.from_numpy(mask_ctx), torch.from_numpy(mask_tgt)

