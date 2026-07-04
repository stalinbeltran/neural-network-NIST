"""Tests de la capa de datos: la estrategia subset es un transform."""
import pytest

torch = pytest.importorskip("torch")

from nnist.data import build_transform


def test_crop_window_reduces_area():
    img = torch.randn(1, 28, 28)
    crop = build_transform("crop_window", top=7, left=7, height=14, width=14)
    out = crop(img)
    assert out.shape == (1, 14, 14)
