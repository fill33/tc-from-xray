"""Valutazione sul test set: errore in HU per regione, coerenza di riproiezione, Dice polmonare
e intervallo di confidenza bootstrap per paziente della differenza A - C.

Nota: la segmentazione polmonare è una soglia sulle HU (pilota). Il piano prevede un
segmentatore pre-addestrato: la sostituzione va registrata come deviazione nel log.

Esempio:
    python -m tcrec.evaluate --cache /kaggle/working/cache --split /kaggle/working/cache/split_pilot40.json \
        --ckpt-a /kaggle/working/runs/A/best.pt --ckpt-c /kaggle/working/runs/C/best.pt --out /kaggle/working/results
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy import ndimage

from .data import make_drr_geometries
from .geometry import project
from .hu import mu_to_hu
from .train import CachedPatients, build_model
from torch.utils.data import DataLoader


def region_masks(hu: np.ndarray) -> dict[str, np.ndarray]:
    """Regioni dalla TC vera: corpo riempito, polmone (aria), tessuto molle, osso."""
    body = ndimage.binary_fill_holes(hu > -300)
    lung = body & (hu < -400)
    soft = body & (hu >= -400) & (hu < 200)
    bone = body & (hu >= 200)
    return {"lung": lung, "soft": soft, "bone": bone, "body": body}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")


def predict(model, loader, dev) -> list[np.ndarray]:
    out = []
    model.eval()
    with torch.no_grad():
        for mu, ap, lat in loader:
            pred = model([ap.to(dev), lat.to(dev)]).cpu().numpy()
            out.extend(list(pred[:, 0]))
    return out


def reproj_error(mu_pred: np.ndarray, ap: np.ndarray, lat: np.ndarray, geoms, spacing: float) -> float:
    vol = torch.from_numpy(mu_pred.astype(np.float32))[None, None]
    with torch.no_grad():
        p_ap = project(vol, geoms[0], spacing, n_samples=128)[0, 0].numpy()
        p_lat = project(vol, geoms[1], spacing, n_samples=128)[0, 0].numpy()
    return float(0.5 * (np.abs(p_ap - ap).mean() + np.abs(p_lat - lat).mean()))


def metrics_for_patient(mu_true: np.ndarray, mu_pred: np.ndarray, ap, lat, geoms, spacing) -> dict:
    hu_true = mu_to_hu(mu_true)
    hu_pred = mu_to_hu(mu_pred)
    masks = region_masks(hu_true)
    row = {}
    for name in ("lung", "soft", "bone", "body"):
        m = masks[name]
        row[f"mae_{name}_hu"] = float(np.abs(hu_pred[m] - hu_true[m]).mean()) if m.any() else float("nan")
    pred_lung = region_masks(hu_pred)["lung"]
    row["dice_lung"] = dice(masks["lung"], pred_lung)
    row["reproj_l1"] = reproj_error(mu_pred, ap, lat, geoms, spacing)
    return row


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]  # pazienti senza la regione sono esclusi, non azzerati
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_model(ckpt: str, shape: int, spacing: float, det_spacing: float, base: int, dev):
    model, geoms = build_model(shape, spacing, det_spacing, base)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    return model.to(dev), geoms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-c", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shape", type=int, default=64)
    p.add_argument("--spacing", type=float, default=6.0)
    p.add_argument("--det-spacing", type=float, default=10.0)
    p.add_argument("--base", type=int, default=16)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.split, encoding="utf-8") as fh:
        split = json.load(fh)
    ids = split["test"]
    os.makedirs(args.out, exist_ok=True)
    loader = DataLoader(CachedPatients(args.cache, ids), batch_size=1)

    results = {}
    for name, ckpt in (("A", args.ckpt_a), ("C", args.ckpt_c)):
        model, geoms = load_model(ckpt, args.shape, args.spacing, args.det_spacing, args.base, dev)
        preds = predict(model, loader, dev)
        rows = []
        for pid, pred, (mu, ap, lat) in zip(ids, preds, [(m, a, l) for m, a, l in loader_items(args.cache, ids)]):
            row = metrics_for_patient(mu, pred, ap, lat, geoms, args.spacing)
            row["patient"] = pid
            row["model"] = name
            rows.append(row)
        results[name] = rows

    csv_path = os.path.join(args.out, "metrics_test.csv")
    keys = list(results["A"][0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for name in ("A", "C"):
            for r in results[name]:
                w.writerow(r)

    summary = {}
    for metric in ("mae_lung_hu", "mae_soft_hu", "mae_bone_hu", "reproj_l1", "dice_lung"):
        summary[metric] = {
            "A": bootstrap_ci(np.array([r[metric] for r in results["A"]])),
            "C": bootstrap_ci(np.array([r[metric] for r in results["C"]])),
        }
    # Criterio di falsificazione pre-registrato: differenza A - C per paziente, MAE in HU.
    diff = {}
    for metric in ("mae_lung_hu", "mae_soft_hu"):
        d = np.array([a[metric] - c[metric] for a, c in zip(results["A"], results["C"])])
        mean, lo, hi = bootstrap_ci(d)
        diff[metric] = {"mean_A_minus_C": mean, "ci95": [lo, hi],
                        "hypothesis_supported": bool(lo > 0)}
    out = {"n_test_patients": len(ids), "summary_ci95": summary, "difference_A_minus_C": diff,
           "note": "campione pilota: intervalli molto ampi, esito indicativo"}
    with open(os.path.join(args.out, "bootstrap_ci.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(diff, indent=2))


def loader_items(cache_dir: str, ids: list[str]):
    ds = CachedPatients(cache_dir, ids)
    for i in range(len(ds)):
        mu, ap, lat = ds[i]
        yield mu[0].numpy(), ap[0].numpy(), lat[0].numpy()


if __name__ == "__main__":
    main()
