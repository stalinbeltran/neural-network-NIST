"""LR range test (Leslie Smith): encuentra un learning rate inicial bueno automáticamente.

Entrena unos pocos batches subiendo el LR de forma geométrica desde `lr_min` a `lr_max` y registra
la loss (suavizada). Al principio, con LR minúsculo, la loss casi no baja; en la zona buena baja
rápido; con LR demasiado alto la loss explota. El LR sugerido es el del **descenso más pronunciado**
(justo antes de que empiece a dispararse). Un solo pase de segundos reemplaza el tuneo manual.

No entrena de verdad: guarda los pesos al empezar y los RESTAURA al terminar, así el modelo queda
como estaba (listo para entrenarse desde cero con el LR encontrado).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .trainer import _make_optimizer


def find_lr(model, train_loader, *, optimizer: str = "adam", weight_decay: float = 0.0,
            device: str = "cpu", lr_min: float = 1e-6, lr_max: float = 1.0,
            num_iter: int = 100, diverge_factor: float = 4.0, beta: float = 0.98) -> dict:
    """Ejecuta el LR range test. Devuelve {'lrs', 'losses', 'suggested_lr'}.

    `num_iter` batches (ciclando el loader si hace falta). Para antes si la loss suavizada supera
    `diverge_factor` veces su mejor valor (ya explotó). `beta` suaviza la loss (EMA con corrección
    de sesgo). El modelo se restaura a su estado inicial al terminar."""
    device = torch.device(device)
    model.to(device)
    init_state = copy.deepcopy(model.state_dict())
    opt = _make_optimizer(optimizer, model.parameters(), lr_min, weight_decay)
    criterion = nn.CrossEntropyLoss()

    gamma = (lr_max / lr_min) ** (1.0 / max(num_iter - 1, 1))   # factor multiplicativo por batch
    lr = lr_min
    avg_loss, best = 0.0, float("inf")
    lrs: list[float] = []
    losses: list[float] = []

    model.train()
    data_iter = iter(train_loader)
    for i in range(num_iter):
        try:
            x, y = next(data_iter)
        except StopIteration:                          # loader agotado -> reciclar
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smooth = avg_loss / (1 - beta ** (i + 1))       # corrección de sesgo del EMA
        lrs.append(lr)
        losses.append(smooth)
        best = min(best, smooth)
        if i > 0 and smooth > diverge_factor * best:    # la loss explotó -> no seguir subiendo
            break
        lr *= gamma

    model.load_state_dict(init_state)                   # el finder NO deja el modelo entrenado
    return {"lrs": lrs, "losses": losses, "suggested_lr": _suggest(lrs, losses)}


def _suggest(lrs: list[float], losses: list[float]) -> float:
    """LR del punto de descenso más pronunciado: gradiente mínimo de la loss vs log10(lr)."""
    if len(lrs) < 3:
        return lrs[len(lrs) // 2] if lrs else 0.0
    log_lrs = [math.log10(l) for l in lrs]
    best_i, best_grad = len(lrs) // 2, float("inf")
    for i in range(1, len(lrs) - 1):
        grad = (losses[i + 1] - losses[i - 1]) / (log_lrs[i + 1] - log_lrs[i - 1])
        if grad < best_grad:
            best_grad, best_i = grad, i
    return lrs[best_i]
