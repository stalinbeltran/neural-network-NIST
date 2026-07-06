"""Transformaciones de entrada, incluida la estrategia SUBSET.

La estrategia "ver un area mas pequeña del digito" NO es un modelo aparte: es una
transformación que recorta/selecciona una ventana de la matriz (CLAUDE.md §5.3).
El modelo se adapta a la forma resultante vía `input_shape` en la config.
"""
from __future__ import annotations

import torch
from torchvision.transforms import functional as TF

from .registry import register


def _clamp01(x):
    """Recorta al rango válido de intensidad [0, 1] tras añadir el ruido."""
    return x.clamp(0.0, 1.0)


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


# ---------------------------------------------------------------------------
# Estrategia RUIDO VISUAL (CLAUDE.md §1, eje de robustez).
# Cada transform degrada la imagen con un tipo de ruido parametrizado. El nombre
# registrado y el nombre del parámetro coinciden con configs/noise/levels.yaml, para
# que `build_transform(tipo, **{param: valor})` funcione directo. Todos operan sobre
# un tensor (C, H, W) en [0, 1] y devuelven otro de la misma forma, recortado a [0, 1].
# El ruido usa el RNG global de torch -> respeta `set_seed` (reproducibilidad, §5.5).
# ---------------------------------------------------------------------------

@register("gaussiano")
def gaussiano(sigma: float):
    """Ruido aditivo gaussiano N(0, sigma) sobre todo el lienzo."""
    def _apply(img):
        return _clamp01(img + torch.randn_like(img) * sigma)
    return _apply


@register("sal_pimienta")
def sal_pimienta(p: float):
    """Ruido impulsivo: fracción p de píxeles forzados a negro/blanco (p/2 cada uno)."""
    def _apply(img):
        m = torch.rand_like(img)
        out = img.clone()
        out[m < p / 2] = 0.0
        out[m > 1 - p / 2] = 1.0
        return out
    return _apply


@register("speckle")
def speckle(sigma: float):
    """Ruido multiplicativo img + img*N(0, sigma): erosiona el trazo, deja el fondo."""
    def _apply(img):
        return _clamp01(img + img * torch.randn_like(img) * sigma)
    return _apply


@register("poisson")
def poisson(scale: float):
    """Shot noise: `scale` = nº de fotones equivalente; MENOR scale = más ruidoso."""
    def _apply(img):
        return _clamp01(torch.poisson(img * scale) / scale)
    return _apply


@register("uniforme")
def uniforme(amplitud: float):
    """Ruido aditivo uniforme en [-amplitud, +amplitud]."""
    def _apply(img):
        return _clamp01(img + (torch.rand_like(img) * 2 - 1) * amplitud)
    return _apply


@register("desenfoque_gaussiano")
def desenfoque_gaussiano(sigma_blur: float):
    """Pérdida de foco (desenfoque gaussiano); kernel impar derivado de sigma."""
    k = max(3, int(2 * round(3 * sigma_blur) + 1) | 1)   # tamaño impar y >= 3
    def _apply(img):
        return TF.gaussian_blur(img, kernel_size=k, sigma=sigma_blur)
    return _apply


@register("oclusion")
def oclusion(lado: int):
    """Tapa (a negro) un bloque cuadrado aleatorio de lado `lado` px."""
    lado = int(lado)
    def _apply(img):
        _, H, W = img.shape
        h, w = min(lado, H), min(lado, W)
        top = int(torch.randint(0, H - h + 1, (1,)).item())
        left = int(torch.randint(0, W - w + 1, (1,)).item())
        out = img.clone()
        out[:, top:top + h, left:left + w] = 0.0
        return out
    return _apply


@register("iluminacion_desigual")
def iluminacion_desigual(amplitud: float):
    """Gradiente de fondo horizontal (foto mal iluminada): suma 0..amplitud de izq. a der."""
    def _apply(img):
        _, _, W = img.shape
        grad = torch.linspace(0.0, amplitud, W, dtype=img.dtype, device=img.device)
        return _clamp01(img + grad.view(1, 1, W))
    return _apply


@register("distorsion_elastica")
def distorsion_elastica(alpha: float, sigma: float = 4.0):
    """Deformación elástica de la forma del dígito (torchvision ElasticTransform)."""
    from torchvision.transforms import ElasticTransform
    et = ElasticTransform(alpha=float(alpha), sigma=float(sigma))
    def _apply(img):
        return _clamp01(et(img))
    return _apply


@register("rayas_horizontales")
def rayas_horizontales(espaciado: int, intensidad: float = 0.5):
    """Patrón periódico: aclara cada `espaciado`-ésima fila. MENOR espaciado = más rayas."""
    espaciado = int(espaciado)
    def _apply(img):
        out = img.clone()
        out[:, ::espaciado, :] = _clamp01(out[:, ::espaciado, :] + intensidad)
        return out
    return _apply


@register("cuantizacion")
def cuantizacion(niveles: int):
    """Reduce la profundidad de bits a `niveles` niveles de gris. MENOS niveles = más degradado."""
    niveles = int(niveles)
    def _apply(img):
        return torch.round(img * (niveles - 1)) / (niveles - 1)
    return _apply


@register("invertido")
def invertido():
    """Negativo fotográfico: invierte la intensidad (pixel p -> 1 - p). Fondo negro <-> trazo blanco.

    No es ruido ni tiene niveles: es una transformación determinista y reversible. Útil para
    estudiar si la red depende de la polaridad fondo/trazo (MNIST es trazo claro sobre fondo negro)."""
    def _apply(img):
        return 1.0 - img
    return _apply
