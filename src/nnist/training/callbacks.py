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


class TrainingLogger(Callback):
    """Actualiza la bitácora `trainings/TRAININGS.md` tras cada época (estado `en_curso`).

    `entry_id` identifica la fila; `total_epochs` es el objetivo para mostrar `hechas/objetivo`.
    Los campos fijos (modelo, datos, checkpoint) se pasan una vez y se conservan en los upserts."""

    def __init__(self, entry_id: str, total_epochs: int, modelo=None, datos=None, checkpoint=None):
        self.entry_id = entry_id
        self.total = total_epochs
        self.modelo = modelo
        self.datos = datos
        self.checkpoint = checkpoint

    def on_epoch_end(self, trainer, epoch, metrics):
        from ..utils import log_training
        log_training(id=self.entry_id, estado="en_curso", modelo=self.modelo, datos=self.datos,
                     checkpoint=self.checkpoint, épocas=f"{trainer._epochs_done}/{self.total}",
                     val=metrics["val_accuracy"])


# TODO: EarlyStopping (parar si val_accuracy no mejora en N épocas).
