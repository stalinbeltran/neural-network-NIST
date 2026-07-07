"""Muestra SOLO las neuronas ganadoras de una red guardada.

Ganadora = neurona que es la mas activa (argmax) para al menos UNA entrada del dataset, con los
pesos actuales. Genera dos imagenes:
  - winners_map.png   : el mapa 50x50, con el campo receptivo de cada ganadora y las perdedoras
                        en negro -> se ve CUANTAS son y DONDE estan.
  - winners_panel.png : solo las ganadoras, ampliadas y ordenadas por nº de entradas que capturan,
                        con etiqueta (indice de neurona y cuantas entradas gana).

Uso:
    python hebbian/show_winners.py --model experiments/<run>/model.npz
    python hebbian/show_winners.py --model experiments/<run>/model.npz --dataset OTRO.npz
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ, SIZE

GRID = 50


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _load_X(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    return X


def winners_map(layer: CompetitiveLayer, winners: np.ndarray) -> Image.Image:
    """Mapa 50x50: campo receptivo de las ganadoras; perdedoras en negro."""
    rf = layer.receptive_fields(SIZE, SIZE)
    gap = 1
    tile = SIZE + gap
    canvas = np.zeros((GRID * tile + gap, GRID * tile + gap), dtype=np.uint8)
    win_set = set(int(i) for i in winners)
    for i in win_set:
        r, c = divmod(i, GRID)
        y, x = gap + r * tile, gap + c * tile
        canvas[y:y + SIZE, x:x + SIZE] = (_norm01(rf[i]) * 255).astype(np.uint8)
    return Image.fromarray(canvas).convert("L")


def winners_panel(layer: CompetitiveLayer, winners: np.ndarray, counts: np.ndarray,
                  up: int = 4) -> Image.Image:
    """Solo las ganadoras, ampliadas, ordenadas por nº de entradas que capturan, con etiqueta."""
    rf = layer.receptive_fields(SIZE, SIZE)
    order = winners[np.argsort(-counts[winners])]          # mas capturas primero
    n = len(order)
    cols = min(n, 8)
    rows = math.ceil(n / cols)
    cell = SIZE * up
    label_h = 14
    pad = 6
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + label_h) + (rows + 1) * pad
    img = Image.new("L", (W, H), 30)
    draw = ImageDraw.Draw(img)
    for k, idx in enumerate(order):
        r, c = divmod(k, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        tile = (_norm01(rf[idx]) * 255).astype(np.uint8)
        tile = Image.fromarray(tile).resize((cell, cell), Image.NEAREST)
        img.paste(tile, (x, y + label_h))
        draw.text((x, y), f"#{int(idx)}  gana {int(counts[idx])}", fill=255)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Muestra solo las neuronas ganadoras de una red guardada")
    ap.add_argument("--model", type=Path, required=True, help="model.npz de una corrida")
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ, help="npz con 'images' para evaluar")
    args = ap.parse_args()

    layer = CompetitiveLayer.load(args.model)
    X = _load_X(args.dataset)
    if layer.n_in != X.shape[1]:
        raise ValueError(f"la red espera {layer.n_in} entradas y el dataset tiene {X.shape[1]}")

    Xn = layer._normalize_rows(X)
    A = layer.activate_batch(Xn)                            # (N, n_out)
    win_of_input = A.argmax(axis=1)
    counts = np.bincount(win_of_input, minlength=layer.n_out)
    winners = np.nonzero(counts)[0]

    out_dir = args.model.parent
    winners_map(layer, winners).save(out_dir / "winners_map.png")
    winners_panel(layer, winners, counts).save(out_dir / "winners_panel.png")

    print(f"neuronas totales : {layer.n_out}")
    print(f"GANADORAS        : {len(winners)}  ({100*len(winners)/layer.n_out:.1f}%)")
    print(f"entradas         : {len(X)}")
    print("reparto (neurona -> nº entradas que gana):")
    for idx in winners[np.argsort(-counts[winners])]:
        print(f"  #{int(idx):4d}  gana {int(counts[idx]):4d} entradas")
    print(f"\n  {out_dir / 'winners_map.png'}    <- mapa 50x50, solo ganadoras (resto en negro)")
    print(f"  {out_dir / 'winners_panel.png'}  <- ganadoras ampliadas + cuantas entradas captura cada una")


if __name__ == "__main__":
    main()
