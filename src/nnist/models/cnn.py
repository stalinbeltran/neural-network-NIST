"""CNN configurable. Preserva la estructura 2D de la imagen (no aplana la entrada).

Cada bloque conv es: Conv2d -> [BatchNorm] -> ReLU -> MaxPool(2). El tamaño espacial se reduce a
la mitad por bloque. Tras los bloques hay una cabeza densa opcional. La dimensión de aplanado se
calcula con un forward de prueba, así funciona con cualquier input_shape (imagen completa o subset).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .registry import register


@register("cnn")
class SimpleCNN(BaseModel):
    def __init__(self, input_shape, num_classes, channels=(16,), kernel_size=3,
                 fc_hidden=128, batchnorm=False, dropout=0.0):
        super().__init__(input_shape, num_classes)

        c_in = input_shape[0]
        conv: list[nn.Module] = []
        for c_out in channels:
            conv.append(nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2))
            if batchnorm:
                conv.append(nn.BatchNorm2d(c_out))
            conv.append(nn.ReLU())
            conv.append(nn.MaxPool2d(2))
            c_in = c_out
        self.features = nn.Sequential(*conv)

        # dimensión de aplanado tras los bloques conv (robusto ante kernel/pool/subset)
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, *input_shape)).numel()

        head: list[nn.Module] = [nn.Flatten()]
        if fc_hidden:
            head += [nn.Linear(n_flat, fc_hidden), nn.ReLU()]
            if dropout:
                head.append(nn.Dropout(dropout))
            head.append(nn.Linear(fc_hidden, num_classes))
        else:
            head.append(nn.Linear(n_flat, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.features(x))
