"""Contrato base para todos los modelos.

Toda arquitectura hereda de `BaseModel`. El nº de parámetros es una métrica de primera
clase del proyecto (ver CLAUDE.md §2), por eso vive aquí.
"""
from __future__ import annotations

import torch.nn as nn


class BaseModel(nn.Module):
    """Base de todas las redes.

    Subclases deben:
      - aceptar `input_shape` (tuple, p. ej. (1, 28, 28) o el recorte del subset) y
        `num_classes` en el constructor, sin hardcodear ninguno de los dos.
      - definir `forward(x)`.
    """

    def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def count_params(self) -> dict[str, int]:
        """Devuelve nº de parámetros totales y entrenables (para el registro de métricas)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"params_total": total, "params_trainable": trainable}

    def forward(self, x):  # pragma: no cover - contrato
        raise NotImplementedError
