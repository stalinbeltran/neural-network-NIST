"""MLP para imagen completa o para un subset (se adapta a `input_shape`)."""
from __future__ import annotations

from math import prod

import torch.nn as nn

from .base import BaseModel
from .registry import register


@register("mlp")
class MLP(BaseModel):
    """Perceptrón multicapa. Aplana la entrada, así que sirve igual para la imagen
    completa que para un recorte (subset): solo cambia `input_shape`."""

    def __init__(self, input_shape, num_classes, hidden=(128,), dropout=0.0):
        super().__init__(input_shape, num_classes)
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = prod(input_shape)
        for h in hidden:
            layers += [nn.Linear(in_features, h), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
