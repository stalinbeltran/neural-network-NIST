"""Registry de transformaciones seleccionables por nombre desde las configs."""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"Transform ya registrada: {name!r}")
        _REGISTRY[name] = fn
        return fn
    return deco


def build_transform(name: str, **kwargs) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Transform desconocida: {name!r}. Registradas: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
