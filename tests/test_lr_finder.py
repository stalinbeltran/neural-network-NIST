"""Tests del LR finder. Sin GPU ni datos reales (CLAUDE.md §9)."""
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from nnist.models import build_model
from nnist.training import find_lr


def _tiny_loader(n=64):
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def test_find_lr_returns_curve_and_suggestion():
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=2, channels=[2, 4], fc_hidden=0)
    res = find_lr(model, _tiny_loader(), num_iter=30, lr_min=1e-5, lr_max=1.0)
    assert len(res["lrs"]) == len(res["losses"]) <= 30
    assert res["lrs"] == sorted(res["lrs"])                 # el lr sube de forma monótona
    assert 1e-5 <= res["suggested_lr"] <= 1.0               # sugerencia dentro del rango sondeado


def test_find_lr_restores_model_weights():
    """El finder sondea pero NO debe dejar el modelo entrenado (restaura los pesos iniciales)."""
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=2, channels=[2, 4], fc_hidden=0)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    find_lr(model, _tiny_loader(), num_iter=30)
    after = model.state_dict()
    assert all(torch.allclose(before[k], after[k]) for k in before)


def test_find_lr_stops_on_divergence():
    """Con lr_max enorme la loss explota -> debe cortar antes de agotar num_iter."""
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=2, channels=[2, 4], fc_hidden=0)
    res = find_lr(model, _tiny_loader(), num_iter=100, lr_min=1e-3, lr_max=1e6, diverge_factor=4.0)
    assert len(res["lrs"]) < 100
