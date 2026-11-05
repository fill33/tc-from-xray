"""Addestramento dei modelli A (lam=0) e C (lam>0) sui volumi preprocessati.

Selezione del checkpoint: sempre sull'errore di volume (L1) sul validation set,
così A e C sono scelti con lo stesso criterio, indipendentemente da lam.

Esempio:
    python -m tcrec.train --cache /kaggle/working/cache --split /kaggle/working/cache/split_pilot40.json \
        --out /kaggle/working/runs/A --lam 0 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import make_drr_geometries
from .losses import total_loss_C, volume_loss
from .models import BackprojUNet


class CachedPatients(Dataset):
    """Legge i file .npz prodotti da tcrec.preprocess."""

    def __init__(self, cache_dir: str, ids: list[str]):
        self.cache_dir = cache_dir
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        d = np.load(os.path.join(self.cache_dir, f"{self.ids[i]}.npz"))
        mu = torch.from_numpy(d["mu"].astype(np.float32))[None]
        ap = torch.from_numpy(d["drr_ap"].astype(np.float32))[None]
        lat = torch.from_numpy(d["drr_lat"].astype(np.float32))[None]
        return mu, ap, lat


def build_model(shape: int, spacing: float, det_spacing: float, base: int) -> tuple[BackprojUNet, tuple]:
    geoms = make_drr_geometries(det_shape=shape, det_spacing=det_spacing)
    model = BackprojUNet(geoms, (shape, shape, shape), spacing, base=base)
    return model, geoms


def run_epoch(model, loader, geoms, args, dev, opt=None) -> tuple[float, float]:
    """Ritorna (loss totale, errore di volume) medi per paziente."""
    total, vol_err, n = 0.0, 0.0, 0
    for mu, ap, lat in loader:
        mu, ap, lat = mu.to(dev), ap.to(dev), lat.to(dev)
        pred = model([ap, lat])
        loss = total_loss_C(pred, mu, [ap, lat], geoms, args.spacing, lam=args.lam)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vol_err += volume_loss(pred, mu).item() * mu.shape[0]
        total += loss.item() * mu.shape[0]
        n += mu.shape[0]
    return total / n, vol_err / n


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.split, encoding="utf-8") as fh:
        split = json.load(fh)
    os.makedirs(args.out, exist_ok=True)

    model, geoms = build_model(args.shape, args.spacing, args.det_spacing, args.base)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(CachedPatients(args.cache, split["train"]), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(CachedPatients(args.cache, split["val"]), batch_size=args.bs)

    best, log = float("inf"), []
    for ep in range(args.epochs):
        model.train()
        tr_loss, _ = run_epoch(model, train_loader, geoms, args, dev, opt=opt)
        model.eval()
        with torch.no_grad():
            va_loss, va_vol = run_epoch(model, val_loader, geoms, args, dev, opt=None)
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "val_volume_l1": va_vol})
        print(f"epoca {ep:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | val_vol_L1 {va_vol:.5f}", flush=True)
        if va_vol < best:
            best = va_vol
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    with open(os.path.join(args.out, "log.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "log": log, "best_val_volume_l1": best}, fh, indent=2)
    print("addestramento completato, miglior val_vol_L1:", best)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lam", type=float, default=0.0, help="0 = modello A; >0 = modello C")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--shape", type=int, default=64)
    p.add_argument("--spacing", type=float, default=6.0)
    p.add_argument("--det-spacing", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
