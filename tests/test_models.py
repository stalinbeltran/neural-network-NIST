"""Tests que no requieren GPU ni descargar datos (CLAUDE.md §9)."""
import pytest

torch = pytest.importorskip("torch")

from nnist.models import build_model


def test_mlp_full_forward_and_param_count():
    model = build_model("mlp", input_shape=(1, 28, 28), num_classes=10, hidden=[128])
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
    counts = model.count_params()
    assert counts["params_total"] > 0


def test_mlp_subset_smaller_input():
    """El mismo MLP con un subset 14x14 debe tener menos parámetros que con 28x28."""
    full = build_model("mlp", input_shape=(1, 28, 28), num_classes=10, hidden=[128])
    subset = build_model("mlp", input_shape=(1, 14, 14), num_classes=10, hidden=[128])
    assert subset.count_params()["params_total"] < full.count_params()["params_total"]
