"""Datos: loaders + transformaciones (incl. estrategia de subset)."""
from . import transforms  # noqa: F401  (efecto: registrar transforms)
from .datasets import (
    DatasetBundle, frozen_stratified_split, load_dataset, load_emnist, load_mnist,
    load_nist_sd19, stratified_split,
)
from .inverted import generate_inverted, inverted_dataset, load_inverted
from .noisy import (
    generate_subset, load_levels, load_noisy_blob, noisy_dataset,
)
from .registry import available, build_transform

__all__ = [
    "DatasetBundle", "load_dataset", "load_mnist", "load_emnist", "load_nist_sd19",
    "stratified_split", "frozen_stratified_split", "build_transform", "available",
    "noisy_dataset", "load_noisy_blob", "generate_subset", "load_levels",
    "inverted_dataset", "load_inverted", "generate_inverted",
]
