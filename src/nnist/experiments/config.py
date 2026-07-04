"""Carga y validación de configs de experimentos (YAML -> dataclasses).

Un experimento se define 100% por config (CLAUDE.md §5.1): dataset, transform (subset),
modelo e hiperparámetros. Aquí se cargan, se resuelve la herencia (`extends`) y se validan.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str
    dataset: dict = field(default_factory=dict)     # {name, split, ...}
    transform: dict | None = None                   # {name: crop_window, top, left, height, width} | None
    model: dict = field(default_factory=dict)       # {name, hidden, ...}
    train: dict = field(default_factory=dict)       # {epochs, lr, batch_size, ...}
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusiona `override` sobre `base` de forma recursiva (los dicts se combinan, el resto pisa)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_raw(path: str | Path) -> dict:
    """Lee un YAML y resuelve `extends` (ruta relativa al propio archivo) recursivamente."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent is not None:
        base = load_raw((path.parent / parent).resolve())
        data = _deep_merge(base, data)
    return data


def from_dict(data: dict) -> ExperimentConfig:
    known = {"name", "dataset", "transform", "model", "train", "seed"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Claves de config desconocidas: {sorted(unknown)}")
    return ExperimentConfig(
        name=data.get("name", "unnamed"),
        dataset=data.get("dataset", {}),
        transform=data.get("transform"),
        model=data.get("model", {}),
        train=data.get("train", {}),
        seed=data.get("seed", 0),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    return from_dict(load_raw(path))


def set_by_path(data: dict, dotted: str, value) -> None:
    """Fija `data['a']['b'] = value` a partir de la clave 'a.b' (usado por los barridos)."""
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
