"""Dataset sintético RECTAS vs CURVAS (2 clases), detrás de la interfaz común de datasets.

Imágenes 28x28 en escala de grises, generadas de forma DETERMINISTA (por semilla):
  - clase 0 = **recta**: un segmento que cruza la imagen con una inclinación dada (0..180°).
  - clase 1 = **curva**: un arco de circunferencia con un radio dado (más pequeño = más curvado).

Se dibuja a 4x y se reduce a 28x28 (antialiasing) para trazos suaves. Se generan `n_per_class`
rectas con inclinaciones repartidas y `n_per_class` curvas con radios repartidos, más un jitter
reproducible de posición/rotación. El pool de TRAIN y el de TEST usan semillas distintas -> son
conjuntos disjuntos (protocolo TEST intocable, CLAUDE.md §3.1). El split train/val se congela igual
que el resto de datasets (`data/splits/lines_curves_*.json`).

  data/processed/shapes/<split>.pt   (dict: images uint8 (N,28,28), labels)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .datasets import DatasetBundle, _split_cache, frozen_stratified_split

SHAPES_ROOT = Path("data/processed/shapes")
SIZE = 28
_SS = 4                       # supersampling para antialiasing del trazo
_S = SIZE * _SS
_WIDTH = 2 * _SS             # ~2 px de grosor tras reducir
CLASSES = {0: "recta", 1: "curva"}


def _downsample(img: Image.Image) -> np.ndarray:
    return np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def _make_line(angle_deg: float, pos_jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Recta CENTRADA que cruza la imagen con inclinación `angle_deg`.

    `pos_jitter` (px) añade desplazamiento aleatorio del centro. Por defecto 0 (baja variabilidad):
    así la ÚNICA variación entre rectas es la inclinación -> pocas imágenes cubren el espacio y una
    red diminuta generaliza. Súbelo para un 'modo difícil' que exija más datos/augmentation."""
    img = Image.new("L", (_S, _S), 0)
    draw = ImageDraw.Draw(img)
    cx = _S / 2 + (rng.uniform(-pos_jitter, pos_jitter) * _SS if pos_jitter else 0.0)
    cy = _S / 2 + (rng.uniform(-pos_jitter, pos_jitter) * _SS if pos_jitter else 0.0)
    ang = np.deg2rad(angle_deg)
    dx, dy = np.cos(ang) * _S, np.sin(ang) * _S     # longitud 2*_S -> cruza toda la imagen
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=255, width=_WIDTH)
    return _downsample(img)


def _make_curve(radius_frac: float, orient_deg: float, span: float,
                pos_jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Arco de circunferencia de radio `radius_frac * _S`, con su punto medio pasando por el centro.

    En baja variabilidad la orientación (`orient_deg`) y la amplitud (`span`) son FIJAS: la única
    variación entre curvas es el radio (curvatura), como pediste. `pos_jitter`/orientación variable
    se dejan como palancas para endurecer la tarea."""
    img = Image.new("L", (_S, _S), 0)
    draw = ImageDraw.Draw(img)
    r = radius_frac * _S
    # centro del círculo tal que el punto (orient_deg) del arco caiga en el centro de la imagen
    ox = rng.uniform(-pos_jitter, pos_jitter) * _SS if pos_jitter else 0.0
    oy = rng.uniform(-pos_jitter, pos_jitter) * _SS if pos_jitter else 0.0
    cx = _S / 2 + ox - np.cos(np.deg2rad(orient_deg)) * r
    cy = _S / 2 + oy - np.sin(np.deg2rad(orient_deg)) * r
    draw.arc([cx - r, cy - r, cx + r, cy + r], orient_deg - span / 2, orient_deg + span / 2,
             fill=255, width=_WIDTH)
    return _downsample(img)


def generate_shapes(split: str, *, n_per_class: int = 30, seed: int = 0,
                    pos_jitter: float = 0.0, rotate_curves: bool = False,
                    root: str | Path = SHAPES_ROOT, save: bool = True) -> dict:
    """Genera (y cachea) un split: `n_per_class` rectas (clase 0) + `n_per_class` curvas (clase 1).

    Baja variabilidad por defecto (`pos_jitter=0`, `rotate_curves=False`): rectas centradas que solo
    varían en inclinación y curvas centradas (pose canónica) que solo varían en radio -> pocas
    imágenes bastan para generalizar. Un pequeño desfase determinista por semilla evita que dos
    splits caigan en los MISMOS ángulos/radios (train y test disjuntos pero de la misma distribución).
    """
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    a_off = rng.uniform(0, 180 / n_per_class)           # desfase de ángulo (train != test)
    for angle in np.linspace(0, 180, n_per_class, endpoint=False):
        imgs.append(_make_line(angle + a_off, pos_jitter, rng))
        labels.append(0)
    r_off = rng.uniform(0, 0.65 / n_per_class)           # desfase de radio
    for radius in np.linspace(0.35, 1.0, n_per_class):
        orient = rng.uniform(0, 360) if rotate_curves else 90.0
        imgs.append(_make_curve(radius + r_off, orient, 140.0, pos_jitter, rng))
        labels.append(1)
    blob = {"images": torch.from_numpy(np.stack(imgs)),
            "labels": torch.tensor(labels, dtype=torch.long), "split": split}
    if save:
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out / f"{split}.pt")
    return blob


def load_shapes_blob(split: str, *, n_per_class: int = 30, seed: int = 0,
                     pos_jitter: float = 0.0, rotate_curves: bool = False,
                     root: str | Path = SHAPES_ROOT, regenerate: bool = False) -> dict:
    """On-demand: reutiliza el `.pt` si existe y tiene el nº de imágenes pedido; si no, (re)genera."""
    path = Path(root) / f"{split}.pt"
    if path.exists() and not regenerate:
        blob = torch.load(path)
        if len(blob["images"]) == 2 * n_per_class:   # caché válido solo si coincide el tamaño
            return blob
    return generate_shapes(split, n_per_class=n_per_class, seed=seed, pos_jitter=pos_jitter,
                           rotate_curves=rotate_curves, root=root)


class _ShapeDataset(Dataset):
    """Imágenes uint8 (N,28,28) -> tensores float (1,28,28) en [0,1]. Expone `.targets` (estratificar)."""

    def __init__(self, images_u8: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images_u8
        self.targets = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i].float().div(255.0).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[i])


def load_lines_curves(transform=None, val_fraction: float = 0.2, split_seed: int = 0,
                      splits_dir: str = "data/splits", n_train_per_class: int = 30,
                      n_test_per_class: int = 100, pos_jitter: float = 0.0,
                      rotate_curves: bool = False, root: str | Path = SHAPES_ROOT) -> DatasetBundle:
    """Bundle RECTAS vs CURVAS con la MISMA interfaz que MNIST (train/val/test, 2 clases).

    El pool de train (por defecto 30+30, lo pedido) se parte train/val con el split congelado; el
    test se genera aparte con otra semilla (disjunto del train). Baja variabilidad por defecto
    (`pos_jitter=0`, `rotate_curves=False`): pocas imágenes bastan para generalizar."""
    # Cada VARIANTE (baja variabilidad vs jitter/rotación) tiene su propio namespace de caché y de
    # split congelado, para que nunca colisionen ni se sirva por error el blob de otra variante.
    variant = "lowvar" if (not pos_jitter and not rotate_curves) else f"pj{pos_jitter:g}_rot{int(rotate_curves)}"
    blob_root = Path(root) / variant

    train_blob = load_shapes_blob("train", n_per_class=n_train_per_class, seed=split_seed,
                                  pos_jitter=pos_jitter, rotate_curves=rotate_curves, root=blob_root)
    test_blob = load_shapes_blob("test", n_per_class=n_test_per_class, seed=split_seed + 1000,
                                 pos_jitter=pos_jitter, rotate_curves=rotate_curves, root=blob_root)

    full_train = _ShapeDataset(train_blob["images"], train_blob["labels"], transform)
    test = _ShapeDataset(test_blob["images"], test_blob["labels"], transform)

    cache = _split_cache(splits_dir, f"lines_curves_{variant}", val_fraction, split_seed)
    train, val = frozen_stratified_split(full_train, val_fraction, split_seed, cache)
    return DatasetBundle(train, val, test, num_classes=2, input_shape=(1, SIZE, SIZE))


# --------------------------------------------------------------------------- rectas H vs V
# Dataset aparte: SEGMENTOS CORTOS clasificados por ORIENTACIÓN. clase 0 = horizontal, 1 = vertical.
# Los segmentos son pequeños y van en posición aleatoria: la señal es la orientación, la posición es
# estorbo. La red debe aprender "solo estas 2 características" (horizontal / vertical).
CLASSES_HV = {0: "horizontal", 1: "vertical"}


def _make_short_line(angle_deg: float, length_frac: float, cx: float, cy: float) -> np.ndarray:
    """Segmento corto de longitud `length_frac*_S` centrado en (cx, cy) con inclinación `angle_deg`."""
    img = Image.new("L", (_S, _S), 0)
    draw = ImageDraw.Draw(img)
    half = length_frac * _S / 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * half, np.sin(a) * half
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=255, width=_WIDTH)
    return _downsample(img)


def generate_lines_hv(split: str, *, n_per_class: int = 40, seed: int = 0, length_frac: float = 0.35,
                      angle_jitter: float = 8.0, root: str | Path = SHAPES_ROOT,
                      save: bool = True) -> dict:
    """`n_per_class` rectas horizontales (clase 0) + `n_per_class` verticales (clase 1), cortas y en
    posición aleatoria. `angle_jitter` da variedad angular sin cruzar los 45° (clases separables)."""
    rng = np.random.default_rng(seed)
    half_px = length_frac * _S / 2
    lo, hi = half_px + _WIDTH, _S - half_px - _WIDTH        # mantener el segmento dentro del lienzo
    imgs, labels = [], []
    for cls, base_angle in ((0, 0.0), (1, 90.0)):
        for _ in range(n_per_class):
            angle = base_angle + rng.uniform(-angle_jitter, angle_jitter)
            cx, cy = rng.uniform(lo, hi), rng.uniform(lo, hi)
            imgs.append(_make_short_line(angle, length_frac, cx, cy))
            labels.append(cls)
    blob = {"images": torch.from_numpy(np.stack(imgs)),
            "labels": torch.tensor(labels, dtype=torch.long), "split": split}
    if save:
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out / f"{split}.pt")
    return blob


def load_hv_blob(split: str, *, n_per_class: int = 40, seed: int = 0, length_frac: float = 0.35,
                 angle_jitter: float = 8.0, root: str | Path = SHAPES_ROOT,
                 regenerate: bool = False) -> dict:
    """On-demand con caché (revalida por tamaño, como el resto de blobs de shapes)."""
    path = Path(root) / f"{split}.pt"
    if path.exists() and not regenerate:
        blob = torch.load(path)
        if len(blob["images"]) == 2 * n_per_class:
            return blob
    return generate_lines_hv(split, n_per_class=n_per_class, seed=seed, length_frac=length_frac,
                             angle_jitter=angle_jitter, root=root)


def load_lines_hv(transform=None, val_fraction: float = 0.2, split_seed: int = 0,
                  splits_dir: str = "data/splits", n_train_per_class: int = 40,
                  n_test_per_class: int = 20, length_frac: float = 0.35, angle_jitter: float = 8.0,
                  root: str | Path = SHAPES_ROOT) -> DatasetBundle:
    """Bundle rectas HORIZONTALES vs VERTICALES (segmentos cortos, posición aleatoria; 2 clases)."""
    variant = f"hv_len{length_frac:g}_jit{angle_jitter:g}"
    blob_root = Path(root) / variant
    train_blob = load_hv_blob("train", n_per_class=n_train_per_class, seed=split_seed,
                              length_frac=length_frac, angle_jitter=angle_jitter, root=blob_root)
    test_blob = load_hv_blob("test", n_per_class=n_test_per_class, seed=split_seed + 1000,
                             length_frac=length_frac, angle_jitter=angle_jitter, root=blob_root)

    full_train = _ShapeDataset(train_blob["images"], train_blob["labels"], transform)
    test = _ShapeDataset(test_blob["images"], test_blob["labels"], transform)

    cache = _split_cache(splits_dir, f"lines_{variant}", val_fraction, split_seed)
    train, val = frozen_stratified_split(full_train, val_fraction, split_seed, cache)
    return DatasetBundle(train, val, test, num_classes=2, input_shape=(1, SIZE, SIZE))


# --------------------------------------------------------------------------- curriculum flexible
# Generador recta (0) vs curva (1) con TODOS los ejes de dificultad parametrizados, para el
# entrenamiento gradual (curriculum): longitud de recta, radio y amplitud de curva, jitter de
# posición, rotación de curvas y ruido gaussiano. Devuelve un blob en memoria (sin caché ni split);
# el orquestador genera datos frescos por etapa. Ver scripts/overnight_curriculum.py.

def _jittered_center(rng: np.random.Generator, jitter: float) -> tuple[float, float]:
    j = jitter * _SS
    return _S / 2 + rng.uniform(-j, j), _S / 2 + rng.uniform(-j, j)


def generate_curriculum(n_per_class: int, seed: int, *, pos_jitter: float = 0.0,
                        len_range: tuple[float, float] = (0.9, 1.1),
                        radius_range: tuple[float, float] = (0.5, 0.7),
                        span_range: tuple[float, float] = (130.0, 150.0),
                        rotate: bool = False, noise: float = 0.0) -> dict:
    """Blob {images uint8 (2N,28,28), labels} con N rectas + N curvas según los rangos de dificultad."""
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for _ in range(n_per_class):                                   # rectas (clase 0)
        length = rng.uniform(*len_range)
        angle = rng.uniform(0, 180)
        cx, cy = _jittered_center(rng, pos_jitter)
        imgs.append(_make_short_line(angle, length, cx, cy))
        labels.append(0)
    for _ in range(n_per_class):                                   # curvas (clase 1)
        r = rng.uniform(*radius_range)
        span = rng.uniform(*span_range)
        orient = rng.uniform(0, 360) if rotate else 90.0
        imgs.append(_make_curve(r, orient, span, pos_jitter, rng))
        labels.append(1)
    images = np.stack(imgs).astype(np.float32)
    if noise > 0:                                                  # ruido gaussiano opcional
        images = images + rng.normal(0, noise * 255, images.shape)
    images = np.clip(images, 0, 255).astype(np.uint8)
    return {"images": torch.from_numpy(images), "labels": torch.tensor(labels, dtype=torch.long)}
