"""Modelos. Importa los módulos para que se registren en el registry."""
from . import cnn, mlp  # noqa: F401  (efecto: registrar arquitecturas)
from .base import BaseModel
from .registry import available, build_model, register

__all__ = ["BaseModel", "build_model", "register", "available"]
