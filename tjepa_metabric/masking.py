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


class MaskIndexSampler:
    """
    Repo-style feature mask sampler (inspired by `.tmp/t-jepa/src/mask.py`):
    - samples a number of context and target features within share ranges
    - draws disjoint context and target index-sets from the feature indices

    This returns indices (not binary masks) so encoders can drop masked features.
    """

    def __init__(
        self,
        n_features: int,
        *,
        min_context_share: float = 0.6,
        max_context_share: float = 0.8,
        min_target_share: float = 0.2,
        max_target_share: float = 0.4,
        allow_overlap: bool = False,
        seed: int = 42,
    ):
        self.n = int(n_features)
        if not (0 < min_context_share < max_context_share <= 1):
            raise ValueError("Expected 0 < min_context_share < max_context_share <= 1")
        if not (0 < min_target_share < max_target_share <= 1):
            raise ValueError("Expected 0 < min_target_share < max_target_share <= 1")
        self.min_ctx = int(round(self.n * float(min_context_share)))
        self.max_ctx = int(round(self.n * float(max_context_share)))
        self.min_tgt = int(round(self.n * float(min_target_share)))
        self.max_tgt = int(round(self.n * float(max_target_share)))
        if self.max_ctx <= self.min_ctx or self.max_tgt <= self.min_tgt:
            raise ValueError("Share ranges are too narrow for this n_features.")
        if self.min_ctx <= 0 or self.min_tgt <= 0:
            raise ValueError("Min context/target must be > 0.")
        self.allow_overlap = bool(allow_overlap)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          - idx_ctx: (B, n_ctx)
          - idx_tgt: (B, n_tgt)
        """
        n_ctx = int(self.rng.integers(self.min_ctx, self.max_ctx + 1))
        n_tgt = int(self.rng.integers(self.min_tgt, self.max_tgt + 1))

        idx_ctx = np.empty((batch_size, n_ctx), dtype=np.int64)
        idx_tgt = np.empty((batch_size, n_tgt), dtype=np.int64)
        all_idx = np.arange(self.n, dtype=np.int64)
        for i in range(batch_size):
            self.rng.shuffle(all_idx)
            ctx = np.sort(all_idx[:n_ctx])
            if self.allow_overlap:
                available = all_idx
            else:
                available = np.setdiff1d(all_idx, ctx, assume_unique=False)
                if available.size < n_tgt:
                    available = all_idx
            self.rng.shuffle(available)
            tgt = np.sort(available[:n_tgt])
            idx_ctx[i] = ctx
            idx_tgt[i] = tgt

        return torch.from_numpy(idx_ctx), torch.from_numpy(idx_tgt)
