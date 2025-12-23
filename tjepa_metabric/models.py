from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MaskBatch:
    x: torch.Tensor  # (B, D)
    idx_ctx: torch.Tensor  # (B, L_ctx) feature indices kept for context
    idx_tgt: torch.Tensor  # (B, L_tgt) feature indices kept for target


def apply_mask_with_indicator(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Feature-space masking like T-JEPA: we provide both masked values and the mask.
    Returns (B, D*2): [x*mask, mask]
    """
    x_masked = x * mask
    return torch.cat([x_masked, mask], dim=1)


class MixedFeatureEmbedder(nn.Module):
    """
    Per-feature embedding layer inspired by T-JEPA:
    - numeric features: per-feature Linear(1 -> token_dim)
    - categorical features: per-feature Embedding(cardinality -> token_dim)

    Produces a (B, n_features, token_dim) tensor aligned to the original feature order.
    """

    def __init__(
        self,
        *,
        n_features: int,
        num_feature_idx: np.ndarray,
        cat_feature_idx: np.ndarray,
        cat_cardinalities: list[int],
        token_dim: int,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.num_feature_idx = np.asarray(num_feature_idx, dtype=int)
        self.cat_feature_idx = np.asarray(cat_feature_idx, dtype=int)
        if len(self.cat_feature_idx) != len(cat_cardinalities):
            raise ValueError("cat_feature_idx and cat_cardinalities must have same length.")
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.token_dim = int(token_dim)

        self._num_map: dict[int, int] = {int(idx): i for i, idx in enumerate(self.num_feature_idx)}
        self._cat_map: dict[int, int] = {int(idx): i for i, idx in enumerate(self.cat_feature_idx)}
        overlap = set(self._num_map).intersection(self._cat_map)
        if overlap:
            raise ValueError(f"Feature(s) declared as both numeric and categorical: {sorted(overlap)}")
        if len(self._num_map) + len(self._cat_map) != self.n_features:
            # We keep this strict so downstream masking stays well-defined.
            missing = sorted(set(range(self.n_features)).difference(self._num_map).difference(self._cat_map))
            raise ValueError(f"num+cat indices must cover all features; missing: {missing}")

        self.num_linears = nn.ModuleList([nn.Linear(1, self.token_dim) for _ in self.num_feature_idx])
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, self.token_dim) for card in self.cat_cardinalities]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expected x of shape (B, {self.n_features}), got {tuple(x.shape)}")

        bsz = x.shape[0]
        out = x.new_zeros((bsz, self.n_features, self.token_dim))
        for feat_idx in range(self.n_features):
            if feat_idx in self._num_map:
                lin = self.num_linears[self._num_map[feat_idx]]
                out[:, feat_idx, :] = lin(x[:, feat_idx : feat_idx + 1])
            else:
                emb = self.cat_embeddings[self._cat_map[feat_idx]]
                card = self.cat_cardinalities[self._cat_map[feat_idx]]
                v = x[:, feat_idx]
                # tolerate float inputs (e.g. 0.0/1.0) and clamp into range
                v = torch.nan_to_num(v, nan=0.0).round().clamp(0, card - 1).to(torch.long)
                out[:, feat_idx, :] = emb(v)
        return out


class FeatureTokenizer(nn.Module):
    """
    T-JEPA-style feature tokenization:
    - numeric: per-feature projection (Linear(1 -> token_dim))
    - categorical: per-feature embedding (Embedding(cardinality -> token_dim))
    - optional feature-type embedding (num vs cat)
    - optional feature-index embedding (column identity)
    """

    def __init__(
        self,
        *,
        n_features: int,
        num_feature_idx: np.ndarray,
        cat_feature_idx: np.ndarray,
        cat_cardinalities: list[int],
        token_dim: int,
        use_feature_type_embedding: bool = True,
        use_feature_index_embedding: bool = True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.token_dim = int(token_dim)
        self.embedder = MixedFeatureEmbedder(
            n_features=self.n_features,
            num_feature_idx=num_feature_idx,
            cat_feature_idx=cat_feature_idx,
            cat_cardinalities=cat_cardinalities,
            token_dim=self.token_dim,
        )

        self.use_feature_type_embedding = bool(use_feature_type_embedding)
        self.use_feature_index_embedding = bool(use_feature_index_embedding)

        if self.use_feature_type_embedding:
            # 0=numeric, 1=categorical
            type_ids = torch.zeros(self.n_features, dtype=torch.long)
            for idx in np.asarray(cat_feature_idx, dtype=int).tolist():
                type_ids[int(idx)] = 1
            self.register_buffer("_feature_type_ids", type_ids, persistent=False)
            self.feature_type_embedding = nn.Embedding(2, self.token_dim)
        else:
            self._feature_type_ids = None
            self.feature_type_embedding = None

        if self.use_feature_index_embedding:
            self.register_buffer(
                "_feature_index_ids",
                torch.arange(self.n_features, dtype=torch.long),
                persistent=False,
            )
            self.feature_index_embedding = nn.Embedding(self.n_features, self.token_dim)
        else:
            self._feature_index_ids = None
            self.feature_index_embedding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedder(x)  # (B, F, H)
        if self.feature_type_embedding is not None:
            t = self.feature_type_embedding(self._feature_type_ids).unsqueeze(0)
            tokens = tokens + t
        if self.feature_index_embedding is not None:
            t = self.feature_index_embedding(self._feature_index_ids).unsqueeze(0)
            tokens = tokens + t
        return tokens


def _gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    tokens: (B, F, H)
    idx:    (B, L)
    returns (B, L, H)
    """
    if idx.dtype != torch.long:
        idx = idx.to(torch.long)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=idx_exp)


class TokenProjector(nn.Module):
    """
    Projection layer (Appendix A.3-style) that maps a token sequence (B, L, H)
    to a fixed vector (B, P) for an MLP.
    """

    def __init__(
        self,
        *,
        mode: str,
        token_dim: int,
        n_tokens: int | None = None,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.mode = str(mode)
        self.token_dim = int(token_dim)
        self.n_tokens = None if n_tokens is None else int(n_tokens)
        self._out_dim = None if out_dim is None else int(out_dim)

        if self.mode in {"mean_pool", "max_pool"}:
            self._out_dim = self.token_dim
            self.proj = None
        elif self.mode == "linear_flatten":
            if self.n_tokens is None or self._out_dim is None:
                raise ValueError("linear_flatten requires n_tokens and out_dim.")
            self.proj = nn.Linear(self.n_tokens * self.token_dim, self._out_dim)
        elif self.mode == "linear_per_feature":
            if self.n_tokens is None:
                raise ValueError("linear_per_feature requires n_tokens.")
            self.proj = nn.ModuleList([nn.Linear(self.token_dim, 1) for _ in range(self.n_tokens)])
            self._out_dim = self.n_tokens
        else:
            raise ValueError(f"Unknown projection mode: {self.mode}")

    @property
    def out_dim(self) -> int:
        if self._out_dim is None:
            raise RuntimeError("out_dim is not initialized.")
        return int(self._out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens of shape (B, L, H), got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {tokens.shape[-1]}")

        if self.mode == "mean_pool":
            return tokens.mean(dim=1)
        if self.mode == "max_pool":
            return tokens.max(dim=1).values
        if self.mode == "linear_flatten":
            if self.n_tokens is None or tokens.shape[1] != self.n_tokens:
                raise ValueError(f"linear_flatten requires exactly L={self.n_tokens} tokens, got {tokens.shape[1]}")
            return self.proj(tokens.reshape(tokens.shape[0], -1))
        if self.mode == "linear_per_feature":
            if self.n_tokens is None or tokens.shape[1] != self.n_tokens:
                raise ValueError(
                    f"linear_per_feature requires exactly L={self.n_tokens} tokens, got {tokens.shape[1]}"
                )
            outs = [self.proj[i](tokens[:, i, :]) for i in range(self.n_tokens)]
            return torch.cat(outs, dim=1)  # (B, L)
        raise RuntimeError("unreachable")


class PaperMLPBackbone(nn.Module):
    """
    Matches the paper repo's MLP blocks:
    - initial Linear
    - (n_hidden-1) blocks of Linear + ReLU + Dropout + BatchNorm1d
    """

    def __init__(self, *, in_dim: int, hidden_dim: int = 256, n_hidden: int = 4, dropout: float = 0.1):
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(n_hidden - 1):
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        return self.blocks(x)


class TokenEncoder(nn.Module):
    """
    Lightweight token encoder (MLP applied token-wise) for tabular tokens.

    This is the smallest MLP-only analog of the transformer encoders used in T-JEPA:
    it preserves the "feature-as-token" representation (B, L, H) and keeps masking
    as *dropping tokens by index*.
    """

    def __init__(
        self,
        *,
        tokenizer: FeatureTokenizer,
        token_dim: int,
        n_reg_tokens: int = 1,
        mlp_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_dim = int(token_dim)
        self.n_reg_tokens = int(n_reg_tokens)
        if self.n_reg_tokens < 0:
            raise ValueError("n_reg_tokens must be >= 0")
        self.reg_tokens = (
            nn.Parameter(torch.zeros(self.n_reg_tokens, self.token_dim))
            if self.n_reg_tokens
            else None
        )

        layers: list[nn.Module] = []
        for _ in range(int(mlp_layers)):
            layers += [
                nn.Linear(self.token_dim, self.token_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        self.token_mlp = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        idx_keep: torch.Tensor | None = None,
        use_reg_tokens: bool = True,
    ) -> torch.Tensor:
        tokens = self.tokenizer(x)  # (B, F, H)
        if idx_keep is not None:
            tokens = _gather_tokens(tokens, idx_keep)  # (B, L, H)
        if use_reg_tokens and self.reg_tokens is not None:
            reg = self.reg_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            tokens = torch.cat([tokens, reg], dim=1)
        return self.token_mlp(tokens)


class TargetTokenPredictor(nn.Module):
    """
    Predict target feature token representations from context tokens.

    We follow the paper's "predict latents, not raw features" principle. To let the
    predictor produce distinct outputs per target feature, we condition on the target
    feature's index (and type) embeddings.
    """

    def __init__(
        self,
        *,
        n_features: int,
        token_dim: int,
        cat_feature_idx: np.ndarray,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_feature_type: bool = True,
        use_feature_index: bool = True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.token_dim = int(token_dim)
        self.use_feature_type = bool(use_feature_type)
        self.use_feature_index = bool(use_feature_index)

        if self.use_feature_index:
            self.feature_index_embedding = nn.Embedding(self.n_features, self.token_dim)
        else:
            self.feature_index_embedding = None

        if self.use_feature_type:
            type_ids = torch.zeros(self.n_features, dtype=torch.long)
            for idx in np.asarray(cat_feature_idx, dtype=int).tolist():
                type_ids[int(idx)] = 1
            self.register_buffer("_feature_type_ids", type_ids, persistent=False)
            self.feature_type_embedding = nn.Embedding(2, self.token_dim)
        else:
            self._feature_type_ids = None
            self.feature_type_embedding = None

        extra = (1 if self.use_feature_index else 0) + (1 if self.use_feature_type else 0)
        in_dim = (1 + extra) * self.token_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.token_dim),
        )

    def forward(self, ctx_tokens: torch.Tensor, idx_tgt: torch.Tensor) -> torch.Tensor:
        if idx_tgt.dtype != torch.long:
            idx_tgt = idx_tgt.to(torch.long)
        # pool context tokens into a single context vector
        ctx = ctx_tokens.mean(dim=1)  # (B, H)
        bsz, n_tgt = idx_tgt.shape
        ctx_rep = ctx.unsqueeze(1).expand(bsz, n_tgt, -1)
        parts = [ctx_rep]
        if self.feature_index_embedding is not None:
            parts.append(self.feature_index_embedding(idx_tgt))
        if self.feature_type_embedding is not None:
            type_ids = self._feature_type_ids[idx_tgt]
            parts.append(self.feature_type_embedding(type_ids))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


class ProjectionLayer(nn.Module):
    """
    Downstream projection layer (Appendix A.3) mapping token matrix R^{d×h}
    into a fixed-size vector used by the supervised model.
    """

    def __init__(
        self,
        *,
        n_features: int,
        token_dim: int,
        mode: str,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.token_dim = int(token_dim)
        self.mode = str(mode)
        self._out_dim = None if out_dim is None else int(out_dim)

        if self.mode == "linear_flatten":
            if self._out_dim is None:
                self._out_dim = self.n_features * self.token_dim
            self.proj = nn.Linear(self.n_features * self.token_dim, self._out_dim)
        elif self.mode in {"linear_per_feature", "mean_pool", "max_pool"}:
            self._out_dim = self.n_features
            if self.mode == "linear_per_feature":
                self.proj = nn.ModuleList([nn.Linear(self.token_dim, 1) for _ in range(self.n_features)])
            else:
                self.proj = None
        else:
            raise ValueError(f"Unknown projection mode: {self.mode}")

    @property
    def out_dim(self) -> int:
        if self._out_dim is None:
            raise RuntimeError("out_dim is not initialized.")
        return int(self._out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.n_features or tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"Expected tokens of shape (B, {self.n_features}, {self.token_dim}), got {tuple(tokens.shape)}"
            )

        if self.mode == "linear_flatten":
            return self.proj(tokens.reshape(tokens.shape[0], -1))
        if self.mode == "mean_pool":
            return tokens.mean(dim=-1)
        if self.mode == "max_pool":
            return tokens.max(dim=-1).values
        if self.mode == "linear_per_feature":
            outs = [self.proj[i](tokens[:, i, :]) for i in range(self.n_features)]
            return torch.cat(outs, dim=1)
        raise RuntimeError("unreachable")


class SurvivalMLP(nn.Module):
    """
    Appendix B.1-style MLP producing a single Cox risk score.
    """

    def __init__(self, *, in_dim: int, hidden_dim: int = 256, n_hidden: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = PaperMLPBackbone(
            in_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            n_hidden=int(n_hidden),
            dropout=float(dropout),
        )
        self.out = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.out(z).squeeze(-1)


class TabularMLPEncoder(nn.Module):
    """
    Mixed-type tabular encoder:
    - tokenizes each feature into a token vector
    - optionally drops masked features by index (context/target subsets)
    - appends [REG] token(s) (pretraining only)
    - projects token sequence to a fixed vector (projection layer)
    - feeds to an MLP backbone (Appendix B.1-style blocks)
    """

    def __init__(
        self,
        *,
        n_features: int,
        num_feature_idx: np.ndarray,
        cat_feature_idx: np.ndarray,
        cat_cardinalities: list[int],
        token_dim: int = 32,
        mlp_hidden_dim: int = 256,
        mlp_layers: int = 4,
        dropout: float = 0.1,
        n_reg_tokens: int = 0,
        use_feature_type_embedding: bool = True,
        use_feature_index_embedding: bool = True,
        projection: str = "mean_pool",
        projection_out_dim: int | None = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.token_dim = int(token_dim)
        self.n_reg_tokens = int(n_reg_tokens)
        self.tokenizer = FeatureTokenizer(
            n_features=self.n_features,
            num_feature_idx=num_feature_idx,
            cat_feature_idx=cat_feature_idx,
            cat_cardinalities=cat_cardinalities,
            token_dim=self.token_dim,
            use_feature_type_embedding=use_feature_type_embedding,
            use_feature_index_embedding=use_feature_index_embedding,
        )
        if self.n_reg_tokens < 0:
            raise ValueError("n_reg_tokens must be >= 0")
        self.reg_tokens = (
            nn.Parameter(torch.zeros(self.n_reg_tokens, self.token_dim))
            if self.n_reg_tokens
            else None
        )

        self.projector = TokenProjector(
            mode=projection,
            token_dim=self.token_dim,
            n_tokens=(self.n_features + self.n_reg_tokens)
            if projection in {"linear_flatten", "linear_per_feature"}
            else None,
            out_dim=projection_out_dim,
        )

        self.mlp = PaperMLPBackbone(
            in_dim=self.projector.out_dim,
            hidden_dim=int(mlp_hidden_dim),
            n_hidden=int(mlp_layers),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor, idx_keep: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.tokenizer(x)  # (B, F, H)
        if idx_keep is not None:
            tokens = _gather_tokens(tokens, idx_keep)
        if self.reg_tokens is not None:
            reg = self.reg_tokens.unsqueeze(0).expand(x.shape[0], -1, -1).to(tokens.dtype)
            tokens = torch.cat([tokens, reg], dim=1)
        z_in = self.projector(tokens)
        return self.mlp(z_in)


class Predictor(nn.Module):
    def __init__(self, emb_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers += [nn.Linear(d, emb_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class JEPA(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        token_dim: int = 32,
        mlp_hidden_dim: int = 256,
        mlp_layers: int = 4,
        pred_hidden: list[int] | None = None,
        dropout: float = 0.1,
        ema: float = 0.996,
        ihc4_feature_idx: np.ndarray | None = None,
        ihc4_aux_weight: float = 5.0,
        num_feature_idx: np.ndarray | None = None,
        cat_feature_idx: np.ndarray | None = None,
        cat_cardinalities: list[int] | None = None,
        n_reg_tokens: int = 3,
        use_feature_type_embedding: bool = True,
        use_feature_index_embedding: bool = True,
        projection: str = "mean_pool",
    ):
        super().__init__()
        pred_hidden = pred_hidden or [512]

        if num_feature_idx is None or cat_feature_idx is None or cat_cardinalities is None:
            # default: all features numeric
            num_feature_idx = np.arange(int(n_features), dtype=int)
            cat_feature_idx = np.array([], dtype=int)
            cat_cardinalities = []

        tokenizer = FeatureTokenizer(
            n_features=int(n_features),
            num_feature_idx=num_feature_idx,
            cat_feature_idx=cat_feature_idx,
            cat_cardinalities=cat_cardinalities,
            use_feature_type_embedding=use_feature_type_embedding,
            use_feature_index_embedding=use_feature_index_embedding,
            token_dim=int(token_dim),
        )

        self.context_encoder = TokenEncoder(
            tokenizer=tokenizer,
            token_dim=int(token_dim),
            n_reg_tokens=int(n_reg_tokens),
            mlp_layers=int(mlp_layers),
            dropout=float(dropout),
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = TargetTokenPredictor(
            n_features=int(n_features),
            token_dim=int(token_dim),
            cat_feature_idx=cat_feature_idx,
            hidden_dim=int(mlp_hidden_dim),
            dropout=float(dropout),
            use_feature_type=use_feature_type_embedding,
            use_feature_index=use_feature_index_embedding,
        )
        self.ema = float(ema)

        self.ihc4_feature_idx = (
            torch.tensor(ihc4_feature_idx, dtype=torch.long)
            if ihc4_feature_idx is not None
            else None
        )
        self.ihc4_aux_weight = float(ihc4_aux_weight)
        if self.ihc4_feature_idx is not None:
            # Auxiliary head to reconstruct IHC4+C features from context embedding
            self.ihc4_head = nn.Linear(int(token_dim), int(self.ihc4_feature_idx.numel()))
        else:
            self.ihc4_head = None

        self.loss = nn.MSELoss()

    @torch.no_grad()
    def update_target_ema(self):
        m = self.ema
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)

    def forward(self, batch: MaskBatch) -> tuple[torch.Tensor, dict[str, float]]:
        # Encode context/target token sets (drop masked features).
        ctx_tokens = self.context_encoder(batch.x, idx_keep=batch.idx_ctx, use_reg_tokens=True)
        with torch.no_grad():
            tgt_tokens = self.target_encoder(batch.x, idx_keep=batch.idx_tgt, use_reg_tokens=True)

        # Strip [REG] from target tokens for the token regression loss.
        n_tgt = int(batch.idx_tgt.shape[1])
        tgt_feat_tokens = tgt_tokens[:, :n_tgt, :]

        pred_tgt_tokens = self.predictor(ctx_tokens, batch.idx_tgt)

        loss_main = self.loss(pred_tgt_tokens, tgt_feat_tokens)
        loss_total = loss_main
        metrics = {"loss_main": float(loss_main.detach().cpu())}

        if self.ihc4_head is not None and self.ihc4_feature_idx is not None:
            y_true = batch.x[:, self.ihc4_feature_idx]
            ctx_pooled = ctx_tokens.mean(dim=1)
            y_pred = self.ihc4_head(ctx_pooled)
            loss_aux = self.loss(y_pred, y_true)
            loss_total = loss_total + self.ihc4_aux_weight * loss_aux
            metrics["loss_ihc4_aux"] = float(loss_aux.detach().cpu())

        metrics["loss_total"] = float(loss_total.detach().cpu())
        return loss_total, metrics


class CoxHead(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z).squeeze(-1)


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cum_risk = torch.logcumsumexp(risk, dim=0)
    return -(risk[event] - log_cum_risk[event]).mean()
