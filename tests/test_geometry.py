"""Test di geometria, retroproiezione, loss e conversione HU.

Questi test non richiedono dati reali: usano volumi sintetici di geometria nota.
"""

import numpy as np
import pytest
import torch

from tcrec.geometry import ViewGeometry, backproject, project
from tcrec.hu import hu_to_mu, mu_to_hu
from tcrec.losses import reprojection_loss, total_loss_C
from tcrec.models import BackprojUNet

SPACING = 4.0  # mm
SHAPE = (24, 24, 24)  # (D, H, W)
AP = ViewGeometry(sod=600.0, sdd=1000.0, det_shape=(24, 24), det_spacing=8.0, lateral=False)
LAT = ViewGeometry(sod=600.0, sdd=1000.0, det_shape=(24, 24), det_spacing=8.0, lateral=True)


def test_empty_volume_projects_to_zero():
    mu = torch.zeros(1, 1, *SHAPE)
    p = project(mu, AP, SPACING)
    assert torch.allclose(p, torch.zeros_like(p), atol=1e-7)


def test_uniform_volume_central_ray_matches_analytic_path_length():
    """Il raggio centrale AP attraversa il volume lungo y: integrale = mu * lunghezza nel volume."""
    c = 0.02  # mm^-1
    mu = torch.full((1, 1, *SHAPE), c)
    p = project(mu, AP, SPACING, n_samples=256)
    nv, nu = AP.det_shape
    center = p[0, 0, nv // 2, nu // 2].item()
    # lunghezza fisica del volume lungo y: (H-1)*spacing; i bordi del campionamento riducono
    # il percorso di circa un voxel, quindi la tolleranza è dichiarata al 10%.
    expected = c * (SHAPE[1] - 1) * SPACING
    assert center == pytest.approx(expected, rel=0.10)


def test_projection_is_linear_in_mu():
    torch.manual_seed(0)
    a = torch.rand(1, 1, *SHAPE)
    b = torch.rand(1, 1, *SHAPE)
    lhs = project(2.0 * a + 3.0 * b, AP, SPACING)
    rhs = 2.0 * project(a, AP, SPACING) + 3.0 * project(b, AP, SPACING)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_projection_gradient_flows():
    mu = torch.rand(1, 1, *SHAPE, requires_grad=True)
    project(mu, LAT, SPACING).sum().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_backprojection_shape_and_zero():
    img = torch.zeros(1, 1, 24, 24)
    out = backproject(img, AP, SPACING, SHAPE)
    assert out.shape == (1, 1, *SHAPE)
    assert torch.allclose(out, torch.zeros_like(out))


def test_backprojection_values_stay_within_image_range():
    """Il campionamento bilineare non deve creare valori fuori dall'intervallo dell'immagine."""
    torch.manual_seed(1)
    img = torch.rand(1, 1, 24, 24)
    out = backproject(img, AP, SPACING, SHAPE)
    inside = out[out != 0]
    assert inside.numel() > 0
    assert inside.min().item() >= img.min().item() - 1e-5
    assert inside.max().item() <= img.max().item() + 1e-5


def test_reprojection_loss_zero_on_consistent_volume():
    torch.manual_seed(2)
    mu = torch.rand(1, 1, *SHAPE)
    targets = [project(mu, AP, SPACING), project(mu, LAT, SPACING)]
    loss = reprojection_loss(mu, targets, (AP, LAT), SPACING)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_total_loss_with_lambda_zero_equals_volume_only():
    torch.manual_seed(3)
    pred = torch.rand(1, 1, *SHAPE)
    target = torch.rand(1, 1, *SHAPE)
    targets = [project(target, AP, SPACING), project(target, LAT, SPACING)]
    lv = torch.nn.functional.l1_loss(pred, target)
    lc = total_loss_C(pred, target, targets, (AP, LAT), SPACING, lam=0.0)
    assert torch.allclose(lv, lc)


def test_model_A_forward_shape_and_nonnegative():
    torch.manual_seed(4)
    model = BackprojUNet((AP, LAT), SHAPE, SPACING, base=4)
    projs = [torch.rand(1, 1, 24, 24), torch.rand(1, 1, 24, 24)]
    out = model(projs)
    assert out.shape == (1, 1, *SHAPE)
    assert (out >= 0).all()


def test_hu_roundtrip_and_air_is_nonnegative():
    hu = np.array([-1000.0, 0.0, 40.0, 1000.0])
    mu = hu_to_mu(hu)
    assert (mu >= 0).all()
    assert np.allclose(mu_to_hu(mu), hu, atol=1e-3)
