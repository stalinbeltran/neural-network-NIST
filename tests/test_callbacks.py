"""Tests de callbacks del Trainer. Sin GPU ni datos (CLAUDE.md §9)."""
import pytest

torch = pytest.importorskip("torch")

from nnist.training import EarlyStopping


class _FakeTrainer:
    """Trainer mínimo para probar el callback: solo necesita `.model` y `.stop_training`."""
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.stop_training = False


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=3, restore_best=False)
    tr = _FakeTrainer()
    # val sube y luego se estanca -> tras 3 épocas sin mejorar debe pedir parar
    for ep, val in enumerate([0.5, 0.9, 0.9, 0.9, 0.9]):
        es.on_epoch_end(tr, ep, {"val_accuracy": val})
    assert tr.stop_training is True


def test_early_stopping_does_not_trigger_while_improving():
    es = EarlyStopping(patience=2)
    tr = _FakeTrainer()
    for ep, val in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
        es.on_epoch_end(tr, ep, {"val_accuracy": val})
    assert tr.stop_training is False


def test_early_stopping_restores_best_weights():
    es = EarlyStopping(patience=2, restore_best=True)
    tr = _FakeTrainer()
    es.on_epoch_end(tr, 0, {"val_accuracy": 0.9})          # mejor -> guarda pesos
    best = tr.model.weight.detach().clone()
    with torch.no_grad():                                   # el modelo "empeora" en épocas siguientes
        tr.model.weight.add_(1.0)
    es.on_epoch_end(tr, 1, {"val_accuracy": 0.8})
    es.on_epoch_end(tr, 2, {"val_accuracy": 0.7})           # 2 sin mejorar -> para y restaura
    assert tr.stop_training is True
    assert torch.allclose(tr.model.weight, best)
