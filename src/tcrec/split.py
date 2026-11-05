"""Divisione dei pazienti in train / validation / test, per paziente (mai per fetta)."""

from __future__ import annotations

import json
import random


def split_by_patient(patient_ids: list[str], fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
                     seed: int = 42) -> dict[str, list[str]]:
    """Divide i pazienti in tre insiemi disgiunti, con seed fissato."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("le frazioni devono sommare a 1")
    ids = sorted(set(patient_ids))
    if len(ids) < 3:
        raise ValueError("servono almeno 3 pazienti")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * fractions[0])))
    n_val = max(1, int(round(n * fractions[1])))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def check_disjoint(split: dict[str, list[str]]) -> None:
    """Solleva errore se un paziente compare in più di un insieme."""
    seen: dict[str, str] = {}
    for name, ids in split.items():
        for pid in ids:
            if pid in seen:
                raise ValueError(f"paziente {pid} in {seen[pid]} e in {name}")
            seen[pid] = name


def save_split(split: dict[str, list[str]], path: str, seed: int, fractions: tuple[float, float, float]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "fractions": list(fractions), **split}, fh, indent=2)
