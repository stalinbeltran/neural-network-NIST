"""Registry de modelos: seleccionar arquitecturas por nombre desde las configs YAML.

Añadir una arquitectura nueva = un archivo en `models/` con `@register("nombre")`.
No hay que tocar el runner ni el sweep (ver CLAUDE.md §5.2).
"""
from __future__ import annotations

from typing import Callable, Type

from .base import BaseModel

_REGISTRY: dict[str, Type[BaseModel]] = {}


def register(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    def deco(cls: Type[BaseModel]) -> Type[BaseModel]:
        if name in _REGISTRY:
            raise ValueError(f"Modelo ya registrado: {name!r}")
        _REGISTRY[name] = cls
        return cls
    return deco


def build_model(name: str, **kwargs) -> BaseModel:
    if name not in _REGISTRY:
        raise KeyError(f"Modelo desconocido: {name!r}. Registrados: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
