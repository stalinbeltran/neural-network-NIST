"""Loaders de datos detrás de una interfaz común.

Se arranca con MNIST/EMNIST (descarga vía torchvision). El diseño deja hueco para NIST SD19
crudo detrás de la MISMA interfaz (CLAUDE.md §3). El nº de clases NO se hardcodea.

`transform` es un callable opcional (p. ej. la ventana de subset de data/transforms.py) que se
aplica sobre el tensor (C, H, W). La `input_shape` reportada es la que ve el modelo, ya recortada.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torchvision import datasets, transforms as T


@dataclass
class DatasetBundle:
    """Lo que un loader devuelve, agnóstico a la fuente."""
    train: object          # torch.utils.data.Dataset
    test: object           # torch.utils.data.Dataset
    num_classes: int
    input_shape: tuple[int, ...]   # forma que recibe el modelo (ya con el subset aplicado)


def _compose(transform: Callable | None):
    steps = [T.ToTensor()]                 # PIL -> tensor (1, 28, 28) en [0, 1]
    if transform is not None:
        steps.append(T.Lambda(transform))  # recorte/ventana de subset sobre el tensor
    return T.Compose(steps)


def _infer_shape(dataset) -> tuple[int, ...]:
    x, _ = dataset[0]
    return tuple(x.shape)


def load_mnist(root: str = "data/raw", transform: Callable | None = None) -> DatasetBundle:
    tf = _compose(transform)
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return DatasetBundle(train, test, num_classes=10, input_shape=_infer_shape(train))


# EMNIST tiene varios splits; cada uno fija el nº de clases (digits=10, letters=26, byclass=62...).
_EMNIST_CLASSES = {"digits": 10, "mnist": 10, "letters": 26, "balanced": 47, "byclass": 62, "bymerge": 47}


def load_emnist(root: str = "data/raw", split: str = "digits",
                transform: Callable | None = None) -> DatasetBundle:
    tf = _compose(transform)
    train = datasets.EMNIST(root, split=split, train=True, download=True, transform=tf)
    test = datasets.EMNIST(root, split=split, train=False, download=True, transform=tf)
    num_classes = _EMNIST_CLASSES.get(split)
    if num_classes is None:
        raise ValueError(f"split EMNIST desconocido: {split!r}")
    return DatasetBundle(train, test, num_classes=num_classes, input_shape=_infer_shape(train))


def load_nist_sd19(root: str = "data/raw/sd19", transform: Callable | None = None) -> DatasetBundle:
    """TODO: NIST Special Database 19 crudo (PNG por clase) detrás de la misma interfaz."""
    raise NotImplementedError("load_nist_sd19 pendiente")


_LOADERS = {"mnist": load_mnist, "emnist": load_emnist, "nist_sd19": load_nist_sd19}


def load_dataset(name: str, transform: Callable | None = None, **kwargs) -> DatasetBundle:
    """Dispatch por nombre desde la config. `kwargs` extra (p. ej. split, root) pasan al loader."""
    if name not in _LOADERS:
        raise KeyError(f"Dataset desconocido: {name!r}. Disponibles: {sorted(_LOADERS)}")
    return _LOADERS[name](transform=transform, **kwargs)
