"""Bucle de entrenamiento/validación, agnóstico al modelo y a la estrategia.

Recibe modelo + dataloaders + hiperparámetros; NO conoce qué arquitectura ni qué subset
se usa (eso lo decide la config). Ver CLAUDE.md §5.4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    optimizer: str = "adam"
    weight_decay: float = 0.0
    device: str = "cpu"


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Optimizador desconocido: {name!r}")


class Trainer:
    def __init__(self, model, train_cfg: TrainConfig, callbacks=None):
        self.model = model
        self.cfg = train_cfg
        self.callbacks = callbacks or []
        self.device = torch.device(train_cfg.device)

    def fit(self, train_loader, val_loader) -> dict:
        """Entrena `epochs` épocas y devuelve el historial de métricas por época."""
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = _make_optimizer(
            self.cfg.optimizer, self.model.parameters(), self.cfg.lr, self.cfg.weight_decay
        )
        history = {"train_loss": [], "val_accuracy": []}
        for epoch in range(self.cfg.epochs):
            self.model.train()
            running = 0.0
            n = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = running / max(n, 1)
            val_acc, _, _, _ = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_accuracy"].append(val_acc)
            metrics = {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc}
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)
        return history

    @torch.no_grad()
    def evaluate(self, loader):
        """Devuelve (accuracy, y_true, y_pred, infer_ms_per_sample) sobre `loader`."""
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []
        t0 = time.perf_counter()
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            y_true.append(y.cpu())
            y_pred.append(preds.cpu())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        accuracy = correct / max(total, 1)
        infer_ms_per_sample = elapsed_ms / max(total, 1)
        yt = torch.cat(y_true) if y_true else torch.empty(0)
        yp = torch.cat(y_pred) if y_pred else torch.empty(0)
        return accuracy, yt, yp, infer_ms_per_sample
