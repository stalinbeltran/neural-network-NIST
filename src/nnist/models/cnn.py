"""CNN sencilla. TODO: implementar. Debe deducir la forma tras conv desde `input_shape`."""
from __future__ import annotations

from .base import BaseModel
from .registry import register


@register("cnn")
class SimpleCNN(BaseModel):
    def __init__(self, input_shape, num_classes, channels=(16, 32)):
        super().__init__(input_shape, num_classes)
        # TODO: construir bloques conv->pool y calcular flatten a partir de input_shape
        raise NotImplementedError("SimpleCNN pendiente de implementar")

    def forward(self, x):  # pragma: no cover
        raise NotImplementedError
