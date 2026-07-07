"""Genera un dataset de RECTAS 28x28 (estilo NIST) para el experimento Hebbiano.

~1000 imágenes en escala de grises, cada una con UNA recta que cruza la imagen con una
inclinacion (0..180 grados) y un pequeno desplazamiento aleatorio del centro. NO hay etiquetas:
el aprendizaje es NO supervisado (competitivo/Hebbiano), asi que solo interesan los pixeles.

Se dibuja a 4x y se reduce a 28x28 (antialiasing), igual que el resto de figuras sinteticas del
proyecto (src/nnist/data/shapes.py). Se guarda como .npz:

    data/processed/lines_hebbian/lines.npz   -> images uint8 (N, 28, 28), angles float (N,)

Uso:
    python hebbian/generate_lines.py                 # 1000 imagenes, semilla 0
    python hebbian/generate_lines.py --n 1000 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 28
_SS = 4                 # supersampling para antialiasing
_S = SIZE * _SS
_WIDTH = 2 * _SS        # ~2 px de grosor tras reducir

OUT = Path("data/processed/lines_hebbian/lines.npz")


def _make_line(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Recta que cruza toda la imagen, centrada en (cx, cy) con inclinacion angle_deg."""
    img = Image.new("L", (_S, _S), 0)
    draw = ImageDraw.Draw(img)
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * _S, np.sin(a) * _S       # longitud 2*_S -> cruza el lienzo entero
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=255, width=_WIDTH)
    small = img.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(small, dtype=np.uint8)


def generate(n: int = 1000, seed: int = 0, pos_jitter: float = 4.0) -> dict:
    """n rectas con angulos repartidos en [0,180) y jitter de posicion reproducible."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 180.0, size=n)
    imgs = np.empty((n, SIZE, SIZE), dtype=np.uint8)
    j = pos_jitter * _SS
    for i, ang in enumerate(angles):
        cx = _S / 2 + rng.uniform(-j, j)
        cy = _S / 2 + rng.uniform(-j, j)
        imgs[i] = _make_line(ang, cx, cy)
    return {"images": imgs, "angles": angles}


def _save_preview(imgs: np.ndarray, path: Path, k: int = 64) -> None:
    """Mosaico PNG con las primeras k imagenes, para inspeccion visual rapida."""
    cols = 8
    rows = (k + cols - 1) // cols
    canvas = np.zeros((rows * SIZE, cols * SIZE), dtype=np.uint8)
    for idx in range(min(k, len(imgs))):
        r, c = divmod(idx, cols)
        canvas[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE] = imgs[idx]
    Image.fromarray(canvas).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera dataset de rectas 28x28 para el experimento Hebbiano")
    ap.add_argument("--n", type=int, default=1000, help="numero de imagenes (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-jitter", type=float, default=4.0, help="desplazamiento max del centro en px")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    blob = generate(args.n, args.seed, args.pos_jitter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=blob["images"], angles=blob["angles"])
    _save_preview(blob["images"], args.out.with_name("preview.png"))

    print(f"OK  {len(blob['images'])} rectas 28x28 -> {args.out}")
    print(f"    preview -> {args.out.with_name('preview.png')}")


if __name__ == "__main__":
    main()
