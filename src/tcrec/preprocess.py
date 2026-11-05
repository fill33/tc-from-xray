"""Preprocessing: da cartelle DICOM a file .npz per paziente (volume mu + due DRR).

Esempio (su Kaggle):
    python -m tcrec.preprocess --data-root /kaggle/input/.../lidc_idri --out /kaggle/working/cache
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np

from .data import compute_drrs, find_ct_series, load_ct_volume, make_drr_geometries, resample_isotropic
from .hu import hu_to_mu


def preprocess_patient(patient_dir: str, out_path: str, target_spacing: float, shape: int,
                       det_spacing: float) -> dict:
    series_dir, files = find_ct_series(patient_dir)
    hu, spacing = load_ct_volume(files)
    hu_cube = resample_isotropic(hu, spacing, target_spacing, shape)
    mu = hu_to_mu(hu_cube).astype(np.float32)
    geoms = make_drr_geometries(det_shape=shape, det_spacing=det_spacing)
    drr_ap, drr_lat = compute_drrs(mu, target_spacing, geoms)
    np.savez_compressed(
        out_path,
        mu=mu.astype(np.float16),
        drr_ap=drr_ap.astype(np.float32),
        drr_lat=drr_lat.astype(np.float32),
        spacing=np.float32(target_spacing),
    )
    return {"series_dir": series_dir, "n_slices": len(files), "orig_spacing": spacing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="cartella con LIDC-IDRI-XXXX")
    ap.add_argument("--out", required=True, help="cartella di output")
    ap.add_argument("--target-spacing", type=float, default=6.0, help="mm")
    ap.add_argument("--shape", type=int, default=64, help="lato del cubo in voxel")
    ap.add_argument("--det-spacing", type=float, default=10.0, help="mm, pixel del rivelatore")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report, errors = {}, {}
    for pid in sorted(os.listdir(args.data_root)):
        pdir = os.path.join(args.data_root, pid)
        if not os.path.isdir(pdir):
            continue
        out_path = os.path.join(args.out, f"{pid}.npz")
        if os.path.exists(out_path):
            report[pid] = "gia presente"
            continue
        try:
            info = preprocess_patient(pdir, out_path, args.target_spacing, args.shape, args.det_spacing)
            report[pid] = info
            print(pid, "OK", info["n_slices"], "fette", flush=True)
        except Exception as exc:  # noqa: BLE001 - registriamo e continuiamo
            errors[pid] = f"{type(exc).__name__}: {exc}"
            print(pid, "ERRORE", errors[pid], flush=True)
            traceback.print_exc()

    with open(os.path.join(args.out, "preprocess_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"ok": {k: v for k, v in report.items() if isinstance(v, dict)},
                   "skipped": [k for k, v in report.items() if isinstance(v, str)],
                   "errors": errors}, fh, indent=2, default=str)
    print("completati:", len([v for v in report.values() if isinstance(v, dict)]), "errori:", len(errors))


if __name__ == "__main__":
    main()
