"""Genera y EXPORTA el dataset sintético rectas vs curvas para poder verlo.

Escribe PNGs individuales y un montaje (contact sheet) por clase en data/processed/shapes/preview/.

Uso:  python scripts/generate_shapes.py            # 30 rectas + 30 curvas (lo pedido)
      python scripts/generate_shapes.py --n 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from nnist.data.shapes import CLASSES, SHAPES_ROOT, generate_shapes


def _montage(images: np.ndarray, cols: int = 10, pad: int = 2, scale: int = 3) -> Image.Image:
    """Rejilla de imágenes (uint8 NxHxW) sobre fondo gris, ampliada para verse bien."""
    n, h, w = images.shape
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad), 60, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    img = Image.fromarray(canvas)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="imágenes por clase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # split propio ("preview") para no pisar el caché de entrenamiento (train.pt), que puede tener
    # un nº de imágenes distinto (escalado para que la CNN generalice).
    blob = generate_shapes("preview", n_per_class=args.n, seed=args.seed)
    images = blob["images"].numpy()
    labels = blob["labels"].numpy()

    preview = Path(SHAPES_ROOT) / "preview"
    for cls, name in CLASSES.items():
        sub = preview / name
        sub.mkdir(parents=True, exist_ok=True)
        cls_imgs = images[labels == cls]
        for i, im in enumerate(cls_imgs):
            Image.fromarray(im).save(sub / f"{name}_{i:02d}.png")
        _montage(cls_imgs).save(preview / f"montage_{name}.png")
        print(f"{len(cls_imgs)} {name}  -> {sub}  (+ montage_{name}.png)")

    print(f"\nMontajes para revisar de un vistazo en: {preview}")


if __name__ == "__main__":
    main()
