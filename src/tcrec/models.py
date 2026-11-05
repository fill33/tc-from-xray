"""Modelli: baseline A (retroproiezione + U-Net 3D) e uscita comune per C."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import ViewGeometry, backproject


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """U-Net 3D a due livelli di profondità. Input (B, cin, D, H, W), output (B, 1, D, H, W)."""

    def __init__(self, cin: int = 2, base: int = 16):
        super().__init__()
        self.enc1 = _block(cin, base)
        self.enc2 = _block(base, base * 2)
        self.bott = _block(base * 2, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv3d(base, 1, 1)
        # Inizializzazione: uscita iniziale vicina a mu dell'acqua (~0.02 mm^-1), non a ~0.7.
        # softplus(b) = 0.02  =>  b = log(exp(0.02) - 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, float(torch.log(torch.expm1(torch.tensor(0.02)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        b = self.bott(F.max_pool3d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        # softplus: il coefficiente di attenuazione è non negativo
        return F.softplus(self.head(d1))


class BackprojUNet(nn.Module):
    """Modello A: retroproietta le due viste su due canali e completa il volume con la U-Net."""

    def __init__(self, geoms: tuple[ViewGeometry, ViewGeometry], vol_shape: tuple[int, int, int],
                 spacing: float, base: int = 16):
        super().__init__()
        self.geoms = geoms
        self.vol_shape = vol_shape
        self.spacing = spacing
        self.net = UNet3D(cin=len(geoms), base=base)

    def backprojections(self, projections: list[torch.Tensor]) -> torch.Tensor:
        vols = [backproject(p, g, self.spacing, self.vol_shape) for p, g in zip(projections, self.geoms)]
        return torch.cat(vols, dim=1)

    def forward(self, projections: list[torch.Tensor]) -> torch.Tensor:
        return self.net(self.backprojections(projections))
