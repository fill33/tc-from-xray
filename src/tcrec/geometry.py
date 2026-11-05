"""Geometria di proiezione a fascio conico e operatori differenziabili.

Convenzioni:
- Volume mu di forma (B, 1, D, H, W) con assi (z, y, x); unità: mm^-1.
- Coordinate fisiche in mm, origine al centro del volume, assi (x, y, z).
- Spaziatura isotropa `spacing` (mm). Estensione fisica: (W, H, D) * spacing.
- Proiezione: integrale di linea di mu lungo ciascun raggio (adimensionale).
  È il termine -log(I/I0) della legge di Beer-Lambert, quindi è lineare in mu.
- Vista AP: sorgente sull'asse y negativo, rivelatore sull'asse y positivo.
  Vista LAT: stessa geometria ruotata di 90 gradi attorno all'asse z.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ViewGeometry:
    """Geometria di una singola vista a fascio conico.

    sod: distanza sorgente-isocentro (mm)
    sdd: distanza sorgente-rivelatore (mm)
    det_shape: (nv, nu) pixel del rivelatore
    det_spacing: spaziatura del pixel del rivelatore (mm)
    lateral: se True la vista è ruotata di 90 gradi attorno a z
    """

    sod: float
    sdd: float
    det_shape: tuple[int, int]
    det_spacing: float
    lateral: bool = False

    def source(self, device=None, dtype=torch.float32) -> torch.Tensor:
        s = torch.zeros(3, device=device, dtype=dtype)
        if self.lateral:
            s[0] = -self.sod
        else:
            s[1] = -self.sod
        return s

    def detector_points(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """Posizioni 3D dei centri dei pixel, shape (nv, nu, 3)."""
        nv, nu = self.det_shape
        d_out = self.sdd - self.sod  # distanza isocentro-rivelatore
        u = (torch.arange(nu, device=device, dtype=dtype) - (nu - 1) / 2) * self.det_spacing
        v = (torch.arange(nv, device=device, dtype=dtype) - (nv - 1) / 2) * self.det_spacing
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pts = torch.zeros(nv, nu, 3, device=device, dtype=dtype)
        if self.lateral:
            pts[..., 0] = d_out
            pts[..., 1] = uu
        else:
            pts[..., 1] = d_out
            pts[..., 0] = uu
        pts[..., 2] = vv
        return pts


def _to_grid(points: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
    """Mappa coordinate fisiche (mm) su [-1, 1] per grid_sample (align_corners=True)."""
    return 2.0 * points / extent


def project(mu: torch.Tensor, geom: ViewGeometry, spacing: float, n_samples: int = 128) -> torch.Tensor:
    """Integrale di linea di mu lungo i raggi della vista. Output (B, 1, nv, nu)."""
    B, C, D, H, W = mu.shape
    assert C == 1, "mu deve avere un solo canale"
    device, dtype = mu.device, mu.dtype
    # align_corners=True: il centro del primo e dell'ultimo voxel corrispondono a -1 e +1
    extent = torch.tensor([(W - 1) * spacing, (H - 1) * spacing, (D - 1) * spacing], device=device, dtype=dtype)

    src = geom.source(device, dtype)                      # (3,)
    det = geom.detector_points(device, dtype)             # (nv, nu, 3)
    nv, nu = geom.det_shape
    ray = det - src                                       # (nv, nu, 3)
    length = ray.norm(dim=-1)                             # (nv, nu) mm

    t = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
    pts = src + t.view(-1, 1, 1, 1) * ray.unsqueeze(0)    # (N, nv, nu, 3)
    grid = _to_grid(pts, extent).unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, N, nv, nu, 3)

    sampled = F.grid_sample(mu, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    # sampled: (B, 1, N, nv, nu); integrale come somma di Riemann sul raggio
    dl = length / n_samples                               # (nv, nu)
    return sampled.sum(dim=2) * dl.unsqueeze(0).unsqueeze(0)


def backproject(image: torch.Tensor, geom: ViewGeometry, spacing: float, vol_shape: tuple[int, int, int]) -> torch.Tensor:
    """Retroproietta un'immagine 2D sulla griglia 3D lungo i raggi della vista.

    Per ogni voxel si calcola il punto in cui la retta sorgente-voxel incontra il
    rivelatore e si campiona l'immagine in quel punto. Output (B, 1, D, H, W).
    """
    D, H, W = vol_shape
    B = image.shape[0]
    device, dtype = image.device, image.dtype

    zs = (torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2) * spacing
    ys = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2) * spacing
    xs = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2) * spacing
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    vox = torch.stack([xx, yy, zz], dim=-1)               # (D, H, W, 3)

    src = geom.source(device, dtype)
    d_out = geom.sdd - geom.sod
    if geom.lateral:
        # piano del rivelatore: x = d_out; coordinate immagine: (y, z)
        denom = (d_out - src[0]) / (vox[..., 0] - src[0])
        hit_u = src[1] + denom * (vox[..., 1] - src[1])   # coordinata y del punto di impatto
    else:
        denom = (d_out - src[1]) / (vox[..., 1] - src[1])
        hit_u = src[0] + denom * (vox[..., 0] - src[0])   # coordinata x del punto di impatto
    hit_v = src[2] + denom * (vox[..., 2] - src[2])       # coordinata z del punto di impatto

    nv, nu = geom.det_shape
    # stessa convenzione di align_corners=True per il rivelatore
    half_u = (nu - 1) * geom.det_spacing / 2
    half_v = (nv - 1) * geom.det_spacing / 2
    gx = hit_u / half_u                                   # normalizzazione su [-1, 1]
    gy = hit_v / half_v
    # grid_sample 2D vuole (B, Hout, Wout, 2); impacchettiamo D e H in Hout.
    grid = torch.stack([gx, gy], dim=-1).reshape(1, D * H, W, 2).expand(B, -1, -1, -1)
    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(B, 1, D, H, W)
