"""Callbacks del Trainer. Se invocan vía `on_epoch_end(trainer, epoch, metrics)`."""
from __future__ import annotations

from pathlib import Path


class Callback:
    def on_epoch_end(self, trainer, epoch, metrics):  # pragma: no cover
        ...


class ModelCheckpoint(Callback):
    """Guarda un checkpoint reanudable cada `every` épocas (y siempre al terminar).

    Escribe SIEMPRE al mismo archivo (sobrescribe el "último"), de modo que en el peor caso solo
    se pierden las épocas desde el último guardado. Con `every=5`, como mucho pierdes 4 épocas.
    El checkpoint incluye pesos + estado del optimizador + época + historial (ver Trainer.state_dict).
    """

    def __init__(self, path, every: int = 5):
        self.path = Path(path)
        self.every = max(1, int(every))

    def on_epoch_end(self, trainer, epoch, metrics):
        completed = epoch + 1
        if completed % self.every == 0 or completed == trainer.cfg.epochs:
            trainer.save_checkpoint(self.path)


# TODO: EarlyStopping (parar si val_accuracy no mejora en N épocas).
