"""Test della pipeline dati su array sintetici (nessun DICOM reale richiesto)."""

import numpy as np
import pytest

from tcrec.data import compute_drrs, make_drr_geometries, resample_isotropic
from tcrec.hu import hu_to_mu
from tcrec.split import check_disjoint, split_by_patient


def test_resample_output_shape_and_air_padding():
    hu = np.full((40, 100, 100), 0.0, dtype=np.float32)  # tessuto a 0 HU
    out = resample_isotropic(hu, spacing=(2.5, 1.0, 1.0), target=5.0, shape=32)
    assert out.shape == (32, 32, 32)
    # la fetta z è più sottile di 5 mm: il volume è più corto, quindi il padding e' aria
    assert out.min() == pytest.approx(-1000.0)


def test_resample_preserves_constant_interior():
    hu = np.full((64, 64, 64), 40.0, dtype=np.float32)
    out = resample_isotropic(hu, spacing=(2.0, 2.0, 2.0), target=4.0, shape=32)
    centre = out[16, 16, 16]
    assert centre == pytest.approx(40.0, abs=1e-3)


def test_drr_of_zero_volume_is_zero():
    mu = np.zeros((32, 32, 32), dtype=np.float32)
    geoms = make_drr_geometries(det_shape=32, det_spacing=10.0)
    drrs = compute_drrs(mu, spacing=6.0, geoms=geoms, n_samples=64)
    assert len(drrs) == 2
    for d in drrs:
        assert d.shape == (32, 32)
        assert np.allclose(d, 0.0)


def test_drr_of_water_cylinder_is_positive_in_centre():
    hu = np.zeros((32, 32, 32), dtype=np.float32)  # acqua
    mu = hu_to_mu(hu)
    geoms = make_drr_geometries(det_shape=32, det_spacing=10.0)
    ap, _ = compute_drrs(mu, spacing=6.0, geoms=geoms, n_samples=64)
    assert ap[16, 16] > 0


def test_split_is_disjoint_and_complete():
    ids = [f"LIDC-IDRI-{i:04d}" for i in range(1, 41)]
    split = split_by_patient(ids, seed=1)
    check_disjoint(split)
    assert sorted(sum(split.values(), [])) == sorted(ids)
    assert len(split["train"]) == 28
    assert len(split["val"]) == 6
    assert len(split["test"]) == 6


def test_split_is_reproducible_with_same_seed():
    ids = [f"LIDC-IDRI-{i:04d}" for i in range(1, 41)]
    assert split_by_patient(ids, seed=7) == split_by_patient(ids, seed=7)


def test_check_disjoint_detects_leak():
    with pytest.raises(ValueError):
        check_disjoint({"train": ["A", "B"], "val": ["B"], "test": ["C"]})
