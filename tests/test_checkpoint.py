"""Tests de checkpoint/resume del Trainer (datos sintéticos, sin descargar MNIST)."""
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from nnist.models import build_model
from nnist.training import ModelCheckpoint, TrainConfig, Trainer


def _loader(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 8, 8, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def test_checkpoint_and_resume(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    train, val = _loader(), _loader(seed=1)

    # entrena 2 épocas guardando cada época
    m1 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    t1 = Trainer(m1, TrainConfig(epochs=2, lr=1e-3), callbacks=[ModelCheckpoint(ckpt, every=1)])
    h1 = t1.fit(train, val)
    assert ckpt.exists()
    assert len(h1["val_accuracy"]) == 2
    assert t1._epochs_done == 2

    # reanuda con un modelo NUEVO hasta 4 épocas
    m2 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    t2 = Trainer(m2, TrainConfig(epochs=4, lr=1e-3), callbacks=[ModelCheckpoint(ckpt, every=1)])
    t2.resume_from(ckpt)
    assert t2.start_epoch == 2

    # los pesos cargados coinciden con los guardados (no es init aleatorio)
    saved = torch.load(ckpt, weights_only=False)["model_state"]
    for k, v in m2.state_dict().items():
        assert torch.allclose(v, saved[k])

    h2 = t2.fit(train, val)
    assert len(h2["val_accuracy"]) == 4      # CONTINUÓ (no reinició desde 0)
    assert t2._epochs_done == 4


def test_resume_rejects_incompatible_model(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    m1 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    Trainer(m1, TrainConfig(epochs=1), callbacks=[ModelCheckpoint(ckpt, every=1)]).fit(_loader(), _loader())

    m2 = build_model("mlp", input_shape=(1, 10, 10), num_classes=3, hidden=[16])  # forma distinta
    with pytest.raises(ValueError):
        Trainer(m2, TrainConfig(epochs=2)).resume_from(ckpt)
