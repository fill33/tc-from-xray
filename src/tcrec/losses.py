"""Loss per il modello A (solo volume) e per il modello C (volume + riproiezione)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import ViewGeometry, project


def volume_loss(pred_mu: torch.Tensor, target_mu: torch.Tensor) -> torch.Tensor:
    """Errore di volume in coefficiente di attenuazione (L1)."""
    return F.l1_loss(pred_mu, target_mu)


def reprojection_loss(pred_mu: torch.Tensor, target_projections: list[torch.Tensor],
                      geoms: tuple[ViewGeometry, ViewGeometry], spacing: float,
                      n_samples: int = 128) -> torch.Tensor:
    """Errore di riproiezione: le proiezioni del volume stimato devono coincidere
    con le proiezioni di ingresso (integrali di linea, legge di Beer-Lambert)."""
    losses = []
    for target, geom in zip(target_projections, geoms):
        pred_p = project(pred_mu, geom, spacing, n_samples=n_samples)
        losses.append(F.l1_loss(pred_p, target))
    return torch.stack(losses).mean()


def total_loss_C(pred_mu: torch.Tensor, target_mu: torch.Tensor, target_projections: list[torch.Tensor],
                 geoms: tuple[ViewGeometry, ViewGeometry], spacing: float, lam: float) -> torch.Tensor:
    """Loss del modello C: L = L_volume + lambda * L_riproiezione. Con lam=0 coincide con A."""
    lv = volume_loss(pred_mu, target_mu)
    if lam == 0.0:
        return lv
    return lv + lam * reprojection_loss(pred_mu, target_projections, geoms, spacing)
