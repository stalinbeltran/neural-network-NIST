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


def _make_line(angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Recta que cruza la imagen con inclinación `angle_deg`, con leve jitter de posición."""
    img = Image.new("L", (_S, _S), 0)
    draw = ImageDraw.Draw(img)
    cx = _S / 2 + rng.uniform(-4, 4) * _SS
    cy = _S / 2 + rng.uniform(-4, 4) * _SS
    ang = np.deg2rad(angle_deg)
    dx, dy = np.cos(ang) * _S, np.sin(ang) * _S     # longitud 2*_S -> cruza toda la imagen
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=255, width=_WIDTH)
    return _downsample(img)


def _make_curve(radius_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Arco de circunferencia de radio `radius_frac * _S`, colocado para pasar por el centro."""
    img = Image.new("L", (_S, _S), 0)
    draw = ImageDraw.Draw(img)
    r = radius_frac * _S
    theta0 = rng.uniform(0, 360)                     # dirección del punto medio del arco
    # centro del círculo tal que el punto (theta0) del arco caiga en el centro de la imagen
    cx = _S / 2 - np.cos(np.deg2rad(theta0)) * r
    cy = _S / 2 - np.sin(np.deg2rad(theta0)) * r
    span = rng.uniform(80, 140)                      # amplitud angular del arco visible
    draw.arc([cx - r, cy - r, cx + r, cy + r], theta0 - span / 2, theta0 + span / 2,
             fill=255, width=_WIDTH)
    return _downsample(img)


def generate_shapes(split: str, *, n_per_class: int = 30, seed: int = 0,
                    root: str | Path = SHAPES_ROOT, save: bool = True) -> dict:
    """Genera (y cachea) un split: `n_per_class` rectas (clase 0) + `n_per_class` curvas (clase 1)."""
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for angle in np.linspace(0, 180, n_per_class, endpoint=False):
        imgs.append(_make_line(angle + rng.uniform(-4, 4), rng))
        labels.append(0)
    for radius in np.linspace(0.35, 1.0, n_per_class):
        imgs.append(_make_curve(radius, rng))
        labels.append(1)
    blob = {"images": torch.from_numpy(np.stack(imgs)),
            "labels": torch.tensor(labels, dtype=torch.long), "split": split}
    if save:
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out / f"{split}.pt")
    return blob


def load_shapes_blob(split: str, *, n_per_class: int = 30, seed: int = 0,
                     root: str | Path = SHAPES_ROOT, regenerate: bool = False) -> dict:
    """On-demand: reutiliza el `.pt` si existe y tiene el nº de imágenes pedido; si no, (re)genera."""
    path = Path(root) / f"{split}.pt"
    if path.exists() and not regenerate:
        blob = torch.load(path)
        if len(blob["images"]) == 2 * n_per_class:   # caché válido solo si coincide el tamaño
            return blob
    return generate_shapes(split, n_per_class=n_per_class, seed=seed, root=root)


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
                      n_test_per_class: int = 20, root: str | Path = SHAPES_ROOT) -> DatasetBundle:
    """Bundle RECTAS vs CURVAS con la MISMA interfaz que MNIST (train/val/test, 2 clases).

    El pool de train (por defecto 30+30, lo pedido) se parte train/val con el split congelado; el
    test se genera aparte con otra semilla (disjunto del train)."""
    train_blob = load_shapes_blob("train", n_per_class=n_train_per_class, seed=split_seed, root=root)
    test_blob = load_shapes_blob("test", n_per_class=n_test_per_class, seed=split_seed + 1000, root=root)

    full_train = _ShapeDataset(train_blob["images"], train_blob["labels"], transform)
    test = _ShapeDataset(test_blob["images"], test_blob["labels"], transform)

    cache = _split_cache(splits_dir, "lines_curves", val_fraction, split_seed)
    train, val = frozen_stratified_split(full_train, val_fraction, split_seed, cache)
    return DatasetBundle(train, val, test, num_classes=2, input_shape=(1, SIZE, SIZE))
