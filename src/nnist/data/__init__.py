"""Datos: loaders + transformaciones (incl. estrategia de subset)."""
from . import transforms  # noqa: F401  (efecto: registrar transforms)
from .datasets import DatasetBundle, load_dataset, load_emnist, load_mnist, load_nist_sd19
from .registry import available, build_transform

__all__ = [
    "DatasetBundle", "load_dataset", "load_mnist", "load_emnist", "load_nist_sd19",
    "build_transform", "available",
]
