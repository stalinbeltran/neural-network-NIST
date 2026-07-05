"""Tests del LR scheduler y de la carga de pesos por posición (datos sintéticos, sin MNIST)."""
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


def test_cosine_scheduler_decays_lr():
    m = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    cfg = TrainConfig(epochs=4, lr=0.01, scheduler="cosine", scheduler_params={"t_max": 4})
    t = Trainer(m, cfg)
    t.fit(_loader(), _loader(seed=1))
    lrs = t.history["lr"]
    assert len(lrs) == 4
    assert lrs[0] > lrs[-1]                       # el lr baja a lo largo del entrenamiento
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # monótonamente no creciente


def test_no_scheduler_keeps_lr_constant():
    m = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    t = Trainer(m, TrainConfig(epochs=3, lr=0.005))
    t.fit(_loader(), _loader(seed=1))
    assert t.history["lr"] == pytest.approx([0.005, 0.005, 0.005])


def test_scheduler_resume_fastforwards_lr(tmp_path):
    """Al reanudar, el lr del scheduler se sitúa donde tocaría (no reinicia en el lr base)."""
    ckpt = tmp_path / "checkpoint.pt"
    cfg = TrainConfig(epochs=2, lr=0.01, scheduler="cosine", scheduler_params={"t_max": 6})
    m1 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    Trainer(m1, cfg, callbacks=[ModelCheckpoint(ckpt, every=1)]).fit(_loader(), _loader(seed=1))

    m2 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    cfg2 = TrainConfig(epochs=6, lr=0.01, scheduler="cosine", scheduler_params={"t_max": 6})
    t2 = Trainer(m2, cfg2)
    t2.resume_from(ckpt)
    t2.fit(_loader(), _loader(seed=1))
    # el historial de lr es continuo (2 previas + 4 nuevas); la 1ª época NUEVA ya viene adelantada
    assert len(t2.history["lr"]) == 6
    assert t2.history["lr"][t2.start_epoch] < 0.01
    assert all(a >= b for a, b in zip(t2.history["lr"], t2.history["lr"][1:]))  # decae
    assert t2._epochs_done == 6


def test_weight_decay_applied_on_resume(tmp_path):
    """Reanudar con un weight_decay distinto lo aplica (no se queda con el del checkpoint)."""
    ckpt = tmp_path / "checkpoint.pt"
    m1 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    Trainer(m1, TrainConfig(epochs=1, lr=0.01, weight_decay=0.0),
            callbacks=[ModelCheckpoint(ckpt, every=1)]).fit(_loader(), _loader(seed=1))

    m2 = build_model("mlp", input_shape=(1, 8, 8), num_classes=3, hidden=[16])
    t2 = Trainer(m2, TrainConfig(epochs=2, lr=0.01, weight_decay=0.05))
    t2.resume_from(ckpt)
    t2.fit(_loader(), _loader(seed=1))
    assert t2.optimizer.param_groups[0]["weight_decay"] == 0.05


def test_resume_by_position_allows_added_dropout(tmp_path):
    """Un modelo entrenado sin dropout se puede reanudar con dropout (misma cantidad de pesos)."""
    ckpt = tmp_path / "checkpoint.pt"
    m1 = build_model("cnn", input_shape=(1, 8, 8), num_classes=3, channels=[8])
    Trainer(m1, TrainConfig(epochs=1, lr=0.01),
            callbacks=[ModelCheckpoint(ckpt, every=1)]).fit(_loader(), _loader(seed=1))

    # con dropout la arquitectura desplaza índices -> carga estricta debe fallar
    m2 = build_model("cnn", input_shape=(1, 8, 8), num_classes=3, channels=[8], dropout=0.3)
    with pytest.raises(RuntimeError):
        Trainer(m2, TrainConfig(epochs=2)).resume_from(ckpt, strict_weights=True)

    # por posición sí carga y los pesos entrenados coinciden con los guardados
    m3 = build_model("cnn", input_shape=(1, 8, 8), num_classes=3, channels=[8], dropout=0.3)
    t3 = Trainer(m3, TrainConfig(epochs=2, lr=0.01))
    t3.resume_from(ckpt, strict_weights=False)
    assert t3.start_epoch == 1
    saved = list(torch.load(ckpt, weights_only=False)["model_state"].values())
    for cur, old in zip(m3.state_dict().values(), saved):
        assert torch.allclose(cur, old)
