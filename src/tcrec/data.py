"""Caricamento delle TC LIDC-IDRI e preparazione dei volumi.

Pipeline per paziente:
1. trova la serie CT (Modality == "CT") con più fette nella cartella del paziente;
2. legge le fette DICOM, applica RescaleSlope/Intercept e ordina per posizione z;
3. ricampiona in isotropo alla spaziatura target e ritaglia/riempie a forma cubica fissa;
4. converte HU in coefficiente di attenuazione mu (mm^-1);
5. genera le due DRR (line integrals) con la geometria di proiezione.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import ViewGeometry, project
from .hu import hu_to_mu


def find_ct_series(patient_dir: str) -> tuple[str, list[str]]:
    """Restituisce (cartella, file) della serie CT con più fette nel paziente."""
    import pydicom

    best: tuple[str, list[str]] | None = None
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(patient_dir, "**", "*"), recursive=True)
                   if os.path.isfile(p)})
    for d in dirs:
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
        if not files:
            continue
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        if best is None or len(files) > len(best[1]):
            best = (d, files)
    if best is None:
        raise FileNotFoundError(f"nessuna serie CT in {patient_dir}")
    return best


def load_ct_volume(files: list[str]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Legge le fette e restituisce (volume in HU con assi z,y,x, spaziatura (z,y,x) in mm)."""
    import pydicom

    slices = [pydicom.dcmread(f) for f in files]
    slices = [s for s in slices if hasattr(s, "ImagePositionPatient") and hasattr(s, "PixelData")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    hu = np.stack([
        s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
        + float(getattr(s, "RescaleIntercept", 0.0))
        for s in slices
    ])
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    dz = float(np.median(np.diff(z))) if len(z) > 1 else float(getattr(slices[0], "SliceThickness", 1.0))
    row_mm, col_mm = (float(v) for v in slices[0].PixelSpacing)  # (riga = y, colonna = x)
    return hu, (abs(dz), row_mm, col_mm)


def resample_isotropic(hu: np.ndarray, spacing: tuple[float, float, float], target: float,
                       shape: int, air_hu: float = -1000.0) -> np.ndarray:
    """Ricampiona a spaziatura isotropa `target` e centra in un cubo shape^3.

    Le parti fuori dal campo di vista vengono riempite con aria (air_hu).
    """
    t = torch.from_numpy(hu)[None, None]
    new_size = [max(1, int(round(n * s / target))) for n, s in zip(hu.shape, spacing)]
    t = F.interpolate(t, size=new_size, mode="trilinear", align_corners=False)[0, 0]
    out = torch.full((shape, shape, shape), air_hu, dtype=t.dtype)
    src_slices, dst_slices = [], []
    for n_src, n_dst in zip(t.shape, out.shape):
        if n_src >= n_dst:
            start = (n_src - n_dst) // 2
            src_slices.append(slice(start, start + n_dst))
            dst_slices.append(slice(0, n_dst))
        else:
            start = (n_dst - n_src) // 2
            src_slices.append(slice(0, n_src))
            dst_slices.append(slice(start, start + n_src))
    out[tuple(dst_slices)] = t[tuple(src_slices)]
    return out.numpy()


def make_drr_geometries(det_shape: int = 64, det_spacing: float = 10.0,
                        sod: float = 600.0, sdd: float = 1000.0) -> tuple[ViewGeometry, ViewGeometry]:
    """Due viste ortogonali (AP e LAT) con la stessa geometria di proiezione."""
    ap = ViewGeometry(sod=sod, sdd=sdd, det_shape=(det_shape, det_shape), det_spacing=det_spacing, lateral=False)
    lat = ViewGeometry(sod=sod, sdd=sdd, det_shape=(det_shape, det_shape), det_spacing=det_spacing, lateral=True)
    return ap, lat


def compute_drrs(mu: np.ndarray, spacing: float, geoms: tuple[ViewGeometry, ViewGeometry],
                 n_samples: int = 128) -> list[np.ndarray]:
    """DRR come integrali di linea di mu (senza rumore). Output: lista di array (nv, nu)."""
    vol = torch.from_numpy(mu.astype(np.float32))[None, None]
    with torch.no_grad():
        return [project(vol, g, spacing, n_samples=n_samples)[0, 0].numpy() for g in geoms]
