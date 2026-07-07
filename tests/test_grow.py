"""Tests del crecimiento gradual (Net2Net). Sin GPU ni datos (CLAUDE.md §9)."""
import pytest

torch = pytest.importorskip("torch")

from nnist.models import build_model, grow_cnn


def _cnn(channels, fc_hidden, batchnorm=False):
    return build_model("cnn", input_shape=(1, 28, 28), num_classes=10,
                       channels=channels, fc_hidden=fc_hidden, batchnorm=batchnorm)


def _assert_preserves(old, new, atol=1e-5):
    """La red crecida debe calcular la MISMA función que la vieja en el instante del trasplante."""
    old.eval(); new.eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(old(x), new(x), atol=atol), "Net2Wider no preservó la salida"


def test_widen_conv_preserves_function():
    old = _cnn([16], fc_hidden=64)
    new = grow_cnn(old, channels=[32], seed=0)
    assert new.channels == (32,)
    assert new.count_params()["params_total"] > old.count_params()["params_total"]
    _assert_preserves(old, new)


def test_widen_fc_preserves_function():
    old = _cnn([16], fc_hidden=64)
    new = grow_cnn(old, fc_hidden=128, seed=1)
    assert new.fc_hidden == 128
    _assert_preserves(old, new)


def test_widen_conv_and_fc_preserves_function():
    old = _cnn([8, 16], fc_hidden=32)
    new = grow_cnn(old, channels=[16, 32], fc_hidden=64, seed=2)
    _assert_preserves(old, new)


def test_widen_with_batchnorm_preserves_function():
    old = _cnn([8], fc_hidden=32, batchnorm=True)
    # BN necesita estadísticas no triviales: hacer unos forwards en train para poblar running_mean/var.
    old.train()
    for _ in range(3):
        old(torch.randn(16, 1, 28, 28))
    new = grow_cnn(old, channels=[20], seed=3)
    _assert_preserves(old, new)


def test_deepen_is_warm_start_and_grows():
    """Añadir un bloque conserva el prefijo (warm-start) y produce una red válida y mayor en conv."""
    old = _cnn([16], fc_hidden=64)
    new = grow_cnn(old, channels=[16, 32], fc_hidden=64, seed=0)
    assert new.channels == (16, 32)
    x = torch.randn(4, 1, 28, 28)
    assert new(x).shape == (4, 10)


def test_grow_rejects_shrinking():
    old = _cnn([32], fc_hidden=64)
    with pytest.raises(ValueError):
        grow_cnn(old, channels=[16], seed=0)


def test_grow_rejects_batchnorm_toggle():
    old = _cnn([16], fc_hidden=64, batchnorm=False)
    with pytest.raises(ValueError):
        grow_cnn(old, channels=[32], batchnorm=True, seed=0)
