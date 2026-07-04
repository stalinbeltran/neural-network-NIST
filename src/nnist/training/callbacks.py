"""Callbacks: early stopping, checkpointing, logging. TODO: implementar."""
from __future__ import annotations


class Callback:
    def on_epoch_end(self, trainer, epoch, metrics):  # pragma: no cover
        ...


# TODO: EarlyStopping, ModelCheckpoint (guarda en experiments/<run_id>/model.pt)
