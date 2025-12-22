from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MaskBatch:
    x: torch.Tensor  # (B, D)
    mask_ctx: torch.Tensor  # (B, D) 1=visible, 0=masked
    mask_tgt: torch.Tensor  # (B, D) 1=visible, 0=masked


def apply_mask_with_indicator(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Feature-space masking like T-JEPA: we provide both masked values and the mask.
    Returns (B, D*2): [x*mask, mask]
    """
    x_masked = x * mask
    return torch.cat([x_masked, mask], dim=1)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers += [nn.Linear(d, emb_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        emb_dim: int = 256,
        enc_hidden: list[int] | None = None,
        pred_hidden: list[int] | None = None,
        dropout: float = 0.1,
        ema: float = 0.996,
        ihc4_feature_idx: np.ndarray | None = None,
        ihc4_aux_weight: float = 5.0,
    ):
        super().__init__()
        enc_hidden = enc_hidden or [512, 512]
        pred_hidden = pred_hidden or [512]

        # +mask indicator doubles feature dim
        self.context_encoder = MLPEncoder(
            in_dim=n_features * 2, emb_dim=emb_dim, hidden=enc_hidden, dropout=dropout
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(emb_dim=emb_dim, hidden=pred_hidden, dropout=dropout)
        self.ema = float(ema)

        self.ihc4_feature_idx = (
            torch.tensor(ihc4_feature_idx, dtype=torch.long)
            if ihc4_feature_idx is not None
            else None
        )
        self.ihc4_aux_weight = float(ihc4_aux_weight)
        if self.ihc4_feature_idx is not None:
            # Auxiliary head to reconstruct IHC4+C features from context embedding
            self.ihc4_head = nn.Linear(emb_dim, int(self.ihc4_feature_idx.numel()))
        else:
            self.ihc4_head = None

        self.loss = nn.MSELoss()

    @torch.no_grad()
    def update_target_ema(self):
        m = self.ema
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)

    def forward(self, batch: MaskBatch) -> tuple[torch.Tensor, dict[str, float]]:
        # Encode context
        x_ctx = apply_mask_with_indicator(batch.x, batch.mask_ctx)
        z_ctx = self.context_encoder(x_ctx)
        z_pred = self.predictor(z_ctx)

        # Encode target (stop-grad)
        with torch.no_grad():
            x_tgt = apply_mask_with_indicator(batch.x, batch.mask_tgt)
            z_tgt = self.target_encoder(x_tgt)

        loss_main = self.loss(z_pred, z_tgt)
        loss_total = loss_main
        metrics = {"loss_main": float(loss_main.detach().cpu())}

        if self.ihc4_head is not None and self.ihc4_feature_idx is not None:
            y_true = batch.x[:, self.ihc4_feature_idx]
            y_pred = self.ihc4_head(z_ctx)
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

