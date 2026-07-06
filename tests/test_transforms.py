"""Tests de los transforms de ruido visual (data/transforms.py).

Sin GPU ni descarga: se usa un tensor pequeño sintético (CLAUDE.md §9).
"""
import torch

from nnist.data import build_transform
from nnist.data.registry import available

# (tipo, params) — nombres y parámetros deben cuadrar con configs/noise/levels.yaml
NOISE_CASES = [
    ("gaussiano", {"sigma": 0.5}),
    ("sal_pimienta", {"p": 0.2}),
    ("speckle", {"sigma": 1.0}),
    ("poisson", {"scale": 15}),
    ("uniforme", {"amplitud": 0.35}),
    ("desenfoque_gaussiano", {"sigma_blur": 1.5}),
    ("oclusion", {"lado": 12}),
    ("iluminacion_desigual", {"amplitud": 0.6}),
    ("distorsion_elastica", {"alpha": 34}),
    ("rayas_horizontales", {"espaciado": 4}),
    ("cuantizacion", {"niveles": 3}),
    ("bajo_contraste", {"factor": 0.40}),
]


def _sample():
    torch.manual_seed(123)
    return torch.rand(1, 28, 28)


def test_all_noise_types_registered():
    for name, _ in NOISE_CASES:
        assert name in available()


def test_shape_and_range_preserved():
    img = _sample()
    for name, params in NOISE_CASES:
        out = build_transform(name, **params)(img)
        assert out.shape == img.shape, name
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, name


def test_noise_is_deterministic_under_same_seed():
    img = _sample()
    for name, params in NOISE_CASES:
        tf = build_transform(name, **params)
        torch.manual_seed(7)
        a = tf(img)
        torch.manual_seed(7)
        b = tf(img)
        assert torch.equal(a, b), name


def test_stochastic_noise_changes_image():
    img = _sample()
    # los aditivos/impulsivos deben modificar la imagen a niveles no triviales
    for name, params in [("gaussiano", {"sigma": 0.5}), ("sal_pimienta", {"p": 0.2})]:
        out = build_transform(name, **params)(img)
        assert not torch.equal(out, img), name


def test_invertido_is_photographic_negative():
    img = _sample()
    out = build_transform("invertido")(img)
    assert out.shape == img.shape
    assert torch.allclose(out, 1.0 - img)
    # doble inversión = identidad
    assert torch.allclose(build_transform("invertido")(out), img)


def test_lazy_subset_generates_caches_and_reuses(tmp_path):
    """On-demand: se genera al primer uso, se cachea y se reutiliza idéntico (determinista)."""
    from nnist.data import load_noisy_blob

    path = tmp_path / "gaussiano" / "nivel_3" / "test.pt"
    assert not path.exists()

    kw = dict(root=tmp_path, limit=16)
    first = load_noisy_blob("gaussiano", "nivel_3", "test", **kw)   # genera + cachea
    assert path.exists()
    assert first["images"].shape == (16, 28, 28)
    assert first["images"].dtype == torch.uint8
    assert first["value"] == 0.50 and first["param"] == "sigma"

    second = load_noisy_blob("gaussiano", "nivel_3", "test", **kw)  # reutiliza el cacheado
    assert torch.equal(first["images"], second["images"])
    assert torch.equal(first["labels"], second["labels"])
