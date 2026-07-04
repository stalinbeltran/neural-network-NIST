"""Métricas multi-objetivo (CLAUDE.md §2).

El rendimiento NO es una sola cifra: accuracy, nº de parámetros, tamaño del área de entrada,
y coste (tiempo/memoria). `RunResult` reúne todo en un registro comparable entre corridas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RunResult:
    run_id: str
    model_name: str
    strategy: str                       # "full" | "subset:<detalle>"
    input_shape: tuple[int, ...]
    num_classes: int
    params_total: int
    params_trainable: int
    accuracy: float = 0.0
    # coste
    train_seconds: float = 0.0
    infer_ms_per_sample: float = 0.0
    # métricas extra (precision/recall/f1 por clase, etc.)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def accuracy(logits, targets) -> float:
    """(argmax(logits) == targets) promedio. `logits`: (N, C); `targets`: (N,)."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def classification_report(y_true, y_pred) -> dict:
    """precision/recall/F1 por clase (dict de sklearn). Requiere scikit-learn."""
    from sklearn.metrics import classification_report as _report

    return _report(y_true.numpy(), y_pred.numpy(), output_dict=True, zero_division=0)
