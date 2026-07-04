"""Transformaciones de entrada, incluida la estrategia SUBSET.

La estrategia "ver un area mas pequeña del digito" NO es un modelo aparte: es una
transformación que recorta/selecciona una ventana de la matriz (CLAUDE.md §5.3).
El modelo se adapta a la forma resultante vía `input_shape` en la config.
"""
from __future__ import annotations

from .registry import register


@register("crop_window")
def crop_window(top: int, left: int, height: int, width: int):
    """Devuelve un callable que recorta una ventana fija [top:top+height, left:left+width].

    Úsalo para la estrategia de subsets: reduces el area vista por la red y comparas
    rendimiento vs. la imagen completa. La `input_shape` del modelo debe ser (C, height, width).
    """
    def _apply(img):  # img: tensor (C, H, W)
        return img[:, top:top + height, left:left + width]
    return _apply


# TODO: otras estrategias de subset — center_crop, patches en rejilla, downsample, etc.
