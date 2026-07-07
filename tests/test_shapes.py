"""Tests del dataset sintético rectas vs curvas. Sin GPU ni descargas (CLAUDE.md §9)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("PIL")

from nnist.data.shapes import SIZE, generate_lines_hv, generate_shapes, load_lines_curves, load_lines_hv


def test_generate_shapes_shapes_and_labels():
    blob = generate_shapes("t", n_per_class=5, seed=0, save=False)
    assert blob["images"].shape == (10, SIZE, SIZE)
    assert blob["labels"].tolist() == [0] * 5 + [1] * 5      # 5 rectas + 5 curvas
    assert blob["images"].max() > 0                          # se dibujó algo (no todo negro)


def test_generate_shapes_deterministic():
    a = generate_shapes("t", n_per_class=4, seed=7, save=False)["images"]
    b = generate_shapes("t", n_per_class=4, seed=7, save=False)["images"]
    assert torch.equal(a, b)


def test_load_lines_curves_bundle(tmp_path):
    bundle = load_lines_curves(n_train_per_class=20, n_test_per_class=10,
                               root=tmp_path / "shapes", splits_dir=str(tmp_path / "splits"))
    assert bundle.num_classes == 2
    assert bundle.input_shape == (1, SIZE, SIZE)
    # el pool de train (40) se parte train/val; el test (20) es disjunto
    assert len(bundle.train) + len(bundle.val) == 40
    assert len(bundle.test) == 20
    x, y = bundle.train[0]
    assert x.shape == (1, SIZE, SIZE)
    assert 0.0 <= float(x.max()) <= 1.0                      # normalizado a [0,1]
    assert y in (0, 1)


def test_generate_lines_hv_labels():
    blob = generate_lines_hv("t", n_per_class=5, seed=0, save=False)
    assert blob["images"].shape == (10, SIZE, SIZE)
    assert blob["labels"].tolist() == [0] * 5 + [1] * 5      # 5 horizontales + 5 verticales
    assert blob["images"].max() > 0


def test_load_lines_hv_bundle(tmp_path):
    bundle = load_lines_hv(n_train_per_class=40, n_test_per_class=20,
                           root=tmp_path / "shapes", splits_dir=str(tmp_path / "splits"))
    assert bundle.num_classes == 2
    assert bundle.input_shape == (1, SIZE, SIZE)
    assert len(bundle.train) + len(bundle.val) == 80         # 40 por clase en el pool de train
    assert len(bundle.test) == 40
