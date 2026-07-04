"""Tests de la capa de datos: transform de subset y split estratificado train/val."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from collections import Counter

from torch.utils.data import Dataset

from nnist.data import build_transform, frozen_stratified_split, stratified_split


def test_crop_window_reduces_area():
    img = torch.randn(1, 28, 28)
    crop = build_transform("crop_window", top=7, left=7, height=14, width=14)
    out = crop(img)
    assert out.shape == (1, 14, 14)


class _FakeDataset(Dataset):
    """Dataset sintético con `.targets` para probar el split sin descargar nada."""
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.zeros(1, 4, 4), self.targets[i]


def test_stratified_split_proportions_and_no_overlap():
    # 100 por clase, 10 clases = 1000 muestras
    targets = [c for c in range(10) for _ in range(100)]
    ds = _FakeDataset(targets)
    train, val = stratified_split(ds, val_fraction=0.2, split_seed=0)

    # tamaños: 20% a validación
    assert len(val) == 200
    assert len(train) == 800

    # sin solape entre train y val
    train_idx = set(train.indices)
    val_idx = set(val.indices)
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx | val_idx) == 1000

    # estratificado: cada clase mantiene la proporción 80/20 (20 por clase en val)
    val_labels = Counter(targets[i] for i in val.indices)
    assert all(count == 20 for count in val_labels.values())


def test_stratified_split_reproducible():
    targets = [c for c in range(10) for _ in range(100)]
    ds = _FakeDataset(targets)
    a, _ = stratified_split(ds, val_fraction=0.2, split_seed=42)
    b, _ = stratified_split(ds, val_fraction=0.2, split_seed=42)
    assert a.indices == b.indices   # misma semilla => mismo split


def test_frozen_split_persists_and_reuses(tmp_path):
    targets = [c for c in range(10) for _ in range(100)]
    ds = _FakeDataset(targets)
    cache = tmp_path / "split.json"

    # 1ª vez: crea el archivo
    train_a, val_a = frozen_stratified_split(ds, 0.2, split_seed=0, cache_path=cache)
    assert cache.exists()

    # 2ª vez: reutiliza EXACTAMENTE los mismos índices desde disco
    train_b, val_b = frozen_stratified_split(ds, 0.2, split_seed=0, cache_path=cache)
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices

    # aunque cambie split_seed, el archivo congelado manda (índices intactos)
    train_c, _ = frozen_stratified_split(ds, 0.2, split_seed=999, cache_path=cache)
    assert train_c.indices == train_a.indices


def test_frozen_split_detects_dataset_change(tmp_path):
    cache = tmp_path / "split.json"
    ds1 = _FakeDataset([c for c in range(10) for _ in range(100)])
    frozen_stratified_split(ds1, 0.2, split_seed=0, cache_path=cache)

    # un dataset distinto (otra huella) contra el mismo cache => error, no split silencioso
    ds2 = _FakeDataset([c for c in range(10) for _ in range(50)])
    with pytest.raises(RuntimeError):
        frozen_stratified_split(ds2, 0.2, split_seed=0, cache_path=cache)
