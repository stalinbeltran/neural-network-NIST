"""Bucle de entrenamiento/validación, agnóstico al modelo y a la estrategia.

Recibe modelo + dataloaders + hiperparámetros; NO conoce qué arquitectura ni qué subset
se usa (eso lo decide la config). Ver CLAUDE.md §5.4.

Soporta CHECKPOINT/RESUME: un checkpoint guarda pesos + estado del optimizador (Adam guarda
momentos) + época completada + historial, de modo que un entrenamiento caro puede reanudarse
sin perder lo ya avanzado. La cadencia (cada N épocas) la fija el callback `ModelCheckpoint`.
Nota: con lr constante (sin scheduler) reanudar es casi equivalente a entrenar seguido; no se
restaura el estado del RNG, así que el orden de barajado tras reanudar puede diferir.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    optimizer: str = "adam"
    weight_decay: float = 0.0                 # regularización L2 (0 = desactivada)
    scheduler: str = "none"                   # "none" | "cosine" | "step" | "plateau"
    scheduler_params: dict = field(default_factory=dict)  # p.ej. {"t_max":20} / {"step_size":5,"gamma":0.1}
    device: str = "cpu"


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Optimizador desconocido: {name!r}")


def _make_scheduler(name: str, optimizer, total_epochs: int, params: dict | None):
    """Crea un LR scheduler por nombre (None si no hay). Baja el lr a lo largo del entrenamiento
    para que el paso sea grande al principio (rápido) y pequeño al final (se asienta sin oscilar)."""
    name = (name or "none").lower()
    if name in ("none", ""):
        return None
    from torch.optim import lr_scheduler as sched
    p = dict(params or {})
    if name == "cosine":       # decae suave (curva coseno) de lr hasta eta_min a lo largo de T_max épocas
        return sched.CosineAnnealingLR(optimizer, T_max=int(p.get("t_max", total_epochs)),
                                       eta_min=float(p.get("eta_min", 0.0)))
    if name == "step":         # multiplica el lr por gamma cada step_size épocas (bajadas en escalón)
        return sched.StepLR(optimizer, step_size=int(p.get("step_size", 5)),
                            gamma=float(p.get("gamma", 0.1)))
    if name == "plateau":      # baja el lr solo cuando val_accuracy deja de mejorar (reactivo)
        return sched.ReduceLROnPlateau(optimizer, mode="max", factor=float(p.get("factor", 0.5)),
                                       patience=int(p.get("patience", 2)))
    raise ValueError(f"Scheduler desconocido: {name!r}")


def _load_state_by_position(model, old_state: dict) -> None:
    """Carga `old_state` en `model` emparejando tensores por ORDEN (no por nombre), validando formas.
    Sirve cuando la arquitectura cambió sin añadir pesos (p.ej. activar dropout renombra
    `classifier.3` -> `classifier.4`, pero los pesos entrenados son los mismos)."""
    new_sd = model.state_dict()
    if len(new_sd) != len(old_state):
        raise ValueError(
            f"No se puede cargar por posición: {len(old_state)} tensores en el checkpoint "
            f"vs {len(new_sd)} en el modelo (¿la arquitectura añadió/quitó pesos?).")
    remapped = {}
    for (nk, nv), (ok, ov) in zip(new_sd.items(), old_state.items()):
        if tuple(nv.shape) != tuple(ov.shape):
            raise ValueError(f"Forma incompatible al cargar por posición: {nk} {tuple(nv.shape)} "
                             f"!= {ok} {tuple(ov.shape)}")
        remapped[nk] = ov
    model.load_state_dict(remapped)


class Trainer:
    def __init__(self, model, train_cfg: TrainConfig, callbacks=None):
        self.model = model
        self.cfg = train_cfg
        self.callbacks = callbacks or []
        self.device = torch.device(train_cfg.device)
        # estado de entrenamiento (mutado en fit; usado por checkpoint/resume)
        self.optimizer = None
        self.scheduler = None
        self.history = {"train_loss": [], "val_accuracy": []}
        self.start_epoch = 0          # nº de épocas ya completadas (>0 si se reanuda)
        self._epochs_done = 0
        self._pending_opt_state = None

    # ---------------------------------------------------------------- checkpoint / resume

    def state_dict(self) -> dict:
        """Todo lo necesario para reanudar: pesos, estado del optimizador, época e historial."""
        return {
            "epochs_done": self._epochs_done,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "history": self.history,
            "train_cfg": asdict(self.cfg),
            "meta": {"input_shape": tuple(self.model.input_shape),
                     "num_classes": self.model.num_classes},
        }

    def save_checkpoint(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def resume_from(self, path, strict_weights: bool = True) -> dict:
        """Carga un checkpoint: restaura pesos + historial + época, y deja pendiente el estado
        del optimizador para aplicarlo al crear el optimizador en `fit`.

        `strict_weights=False` permite cargar en un modelo cuya arquitectura cambió SIN añadir
        pesos (p.ej. activar dropout, que solo desplaza los índices de las capas del `Sequential`):
        se cargan los tensores por posición si todas las formas coinciden."""
        ckpt = torch.load(path, map_location=self.device)
        meta = ckpt.get("meta", {})
        if meta.get("input_shape") and tuple(meta["input_shape"]) != tuple(self.model.input_shape):
            raise ValueError(
                f"Checkpoint incompatible: input_shape {meta['input_shape']} != {self.model.input_shape}")
        if meta.get("num_classes") and meta["num_classes"] != self.model.num_classes:
            raise ValueError(
                f"Checkpoint incompatible: num_classes {meta['num_classes']} != {self.model.num_classes}")
        if strict_weights:
            self.model.load_state_dict(ckpt["model_state"])
        else:
            _load_state_by_position(self.model, ckpt["model_state"])
        self.history = ckpt.get("history", {"train_loss": [], "val_accuracy": []})
        self.start_epoch = ckpt.get("epochs_done", 0)
        self._epochs_done = self.start_epoch
        self._pending_opt_state = ckpt.get("optimizer_state")
        return ckpt

    # ---------------------------------------------------------------- entrenamiento

    def fit(self, train_loader, val_loader) -> dict:
        """Entrena hasta `cfg.epochs` (continúa desde `start_epoch` si se reanudó)."""
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        self.optimizer = _make_optimizer(
            self.cfg.optimizer, self.model.parameters(), self.cfg.lr, self.cfg.weight_decay
        )
        if self._pending_opt_state is not None:      # restaurar momentos de Adam al reanudar
            self.optimizer.load_state_dict(self._pending_opt_state)
            self._pending_opt_state = None
            # la config es la fuente de verdad: reaplica lr/weight_decay actuales (load_state_dict
            # del optimizador restaura los del checkpoint, así que un cambio de wd se ignoraría).
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.lr
                g["weight_decay"] = self.cfg.weight_decay

        # scheduler opcional. Cosine/step son función determinista de la época: al reanudar se
        # "adelanta" start_epoch pasos para situar el lr donde toca. Plateau (reactivo) arranca de cero.
        self.scheduler = _make_scheduler(self.cfg.scheduler, self.optimizer, self.cfg.epochs,
                                         self.cfg.scheduler_params)
        is_plateau = self.scheduler.__class__.__name__ == "ReduceLROnPlateau"
        if self.scheduler is not None and not is_plateau and self.start_epoch > 0:
            import warnings
            with warnings.catch_warnings():   # el "step antes de optimizer.step" aquí es intencional
                warnings.simplefilter("ignore")
                for _ in range(self.start_epoch):
                    self.scheduler.step()
        self.history.setdefault("lr", [])            # back-compat con checkpoints sin esta clave

        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.model.train()
            running = 0.0
            n = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = running / max(n, 1)
            val_acc, _, _, _ = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])  # lr usado en esta época
            self._epochs_done = epoch + 1
            metrics = {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc}
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)
            if self.scheduler is not None:           # avanzar el lr para la siguiente época
                self.scheduler.step(val_acc) if is_plateau else self.scheduler.step()
        return self.history

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
