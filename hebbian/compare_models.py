"""Compara varios checkpoints de la red (p. ej. 10 vs 20 vs 30 epocas): para cada entrada muestra
QUE NEURONAS DISPARAN en cada modelo, lado a lado, para ver como cambia con mas entrenamiento.

Usa un MISMO umbral de disparo theta para todos (asi la diferencia es por los pesos, no por theta).
Genera un montaje: filas = entradas de muestra, columnas = [entrada | modelo1 | modelo2 | ...],
cada celda es el mapa 50x50 digital (blanco = dispara, rojo = ganadora). Ademas imprime estadisticas
(media de neuronas disparadas por entrada en cada modelo y solape entre checkpoints).

Uso:
    python hebbian/compare_models.py --models experiments/inhib_series/model_ep010.npz experiments/inhib_series/model_ep020.npz
    python hebbian/compare_models.py --models A.npz B.npz C.npz --threshold 0.40 --samples 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ, SIZE

GRID = 50


def load_X(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    return X


def firing_rgb(layer: CompetitiveLayer, xu: np.ndarray, thr: float):
    """Mapa 50x50 RGB: blanco = dispara (activacion>=thr), rojo = ganadora. Devuelve (img, fired_mask)."""
    a = layer.W @ xu
    fired = a >= thr
    img = np.zeros((GRID, GRID, 3), dtype=np.uint8)
    img[fired.reshape(GRID, GRID)] = (255, 255, 255)
    w = int(a.argmax())
    img[w // GRID, w % GRID] = (255, 40, 40)
    return img, fired


def up(img: np.ndarray, cell: int) -> Image.Image:
    mode = "RGB" if img.ndim == 3 else "L"
    return Image.fromarray(img).convert(mode).resize((cell, cell), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara neuronas disparadas por entrada entre checkpoints")
    ap.add_argument("--models", type=Path, nargs="+", required=True, help="lista de model.npz a comparar")
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ)
    ap.add_argument("--threshold", type=float, default=0.40, help="umbral de disparo comun a todos")
    ap.add_argument("--samples", type=int, default=8, help="nº de entradas de muestra en el montaje")
    ap.add_argument("--out", type=Path, default=Path("experiments/inhib_series/compare.png"))
    args = ap.parse_args()

    layers = [CompetitiveLayer.load(m) for m in args.models]
    labels = [f"{L.epochs_trained} ep" for L in layers]
    X = load_X(args.dataset)
    Xn = layers[0]._normalize_rows(X)
    thr = args.threshold

    # ---- estadisticas globales sobre TODO el dataset ----
    print(f"umbral de disparo theta = {thr}   entradas = {len(X)}")
    fired_all = []
    for L, lab, m in zip(layers, labels, args.models):
        A = Xn @ L.W.T
        F = A >= thr
        fired_all.append(F)
        print(f"  {lab:>7}  ({m.name})  media disparadas/entrada = {F.sum(1).mean():6.1f}"
              f"   neuronas que disparan alguna vez = {int((F.any(0)).sum())}")
    for i in range(1, len(layers)):
        inter = (fired_all[i - 1] & fired_all[i]).sum(1)
        union = (fired_all[i - 1] | fired_all[i]).sum(1)
        jac = np.divide(inter, union, out=np.zeros_like(inter, float), where=union > 0).mean()
        print(f"  solape (Jaccard) {labels[i-1]} vs {labels[i]}: {jac:.3f}  (1=identico, 0=sin solape)")

    # ---- montaje visual: filas = entradas, columnas = [entrada | modelo1 | ...] ----
    idx = np.linspace(0, len(X) - 1, args.samples).astype(int)
    cell, pad, header, lab_w = 150, 6, 22, 60
    ncol = 1 + len(layers)
    W = lab_w + ncol * cell + (ncol + 1) * pad
    H = header + args.samples * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((lab_w + pad, 6), "entrada", fill=(200, 200, 200))
    for c, lab in enumerate(labels):
        draw.text((lab_w + (c + 1) * (cell + pad) + pad, 6), lab, fill=(150, 230, 150))

    for r, i in enumerate(idx):
        y = header + r * (cell + pad)
        draw.text((4, y + cell // 2), f"#{i}", fill=(160, 160, 160))
        inp = (X[i].reshape(SIZE, SIZE) * 255).astype(np.uint8)
        canvas.paste(up(inp, cell), (lab_w + pad, y))
        for c, L in enumerate(layers):
            img, fired = firing_rgb(L, Xn[i], thr)
            canvas.paste(up(img, cell), (lab_w + (c + 1) * (cell + pad) + pad, y))
            draw.text((lab_w + (c + 1) * (cell + pad) + pad + 3, y + 3),
                      f"{int(fired.sum())}", fill=(255, 220, 120))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\nmontaje -> {args.out}  (blanco=dispara, rojo=ganadora; el nº amarillo = cuantas disparan)")


if __name__ == "__main__":
    main()
