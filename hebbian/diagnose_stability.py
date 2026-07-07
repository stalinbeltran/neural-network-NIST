"""Diagnostico fino de estabilidad entre snapshots consecutivos, para distinguir REORGANIZACION
real de FRAGILIDAD de medida (empates cerca del umbral o del argmax).

Por cada par consecutivo reporta:
  - dW_rel   : cambio relativo de pesos ||Wb-Wa|| / ||Wa||  (0 = pesos sin cambiar).
  - act_cos  : coseno medio, por entrada, entre el vector de activacion de las 2500 neuronas en a y
               en b. Alto = el PATRON de respuesta se conserva (aunque el argmax salte).
  - top10_jac: solape Jaccard de las 10 neuronas mas activas por entrada (robusto a empates).
  - win_match: fraccion de entradas con la MISMA ganadora (argmax), la medida mas estricta.

Uso:
    python hebbian/diagnose_stability.py --dir experiments/series_posneg_anneal \
        --dataset data/processed/lines_hebbian/lines_posneg.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from competitive_net import CompetitiveLayer


def load_X(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    return X


def topk_sets(A: np.ndarray, k: int) -> list[set]:
    idx = np.argpartition(-A, k, axis=1)[:, :k]
    return [set(row) for row in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostico fino de estabilidad entre snapshots")
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    paths = sorted(args.dir.glob("model_ep*.npz"))
    layers = [CompetitiveLayer.load(p) for p in paths]
    eps = [L.epochs_trained for L in layers]
    X = load_X(args.dataset)
    Xn = layers[0]._normalize_rows(X)

    As = [Xn @ L.W.T for L in layers]
    topk = [topk_sets(A, args.k) for A in As]
    wins = [A.argmax(1) for A in As]

    print(f"snapshots {eps}   dataset {args.dataset} ({len(X)})   k={args.k}\n")
    print(f"{'par':>12} | {'dW_rel':>7} | {'act_cos':>7} | {'top'+str(args.k)+'_jac':>8} | {'win_match':>9}")
    for i in range(1, len(layers)):
        Wa, Wb = layers[i - 1].W, layers[i].W
        dW = float(np.linalg.norm(Wb - Wa) / np.linalg.norm(Wa))
        Aa, Ab = As[i - 1], As[i]
        num = (Aa * Ab).sum(1)
        den = np.linalg.norm(Aa, axis=1) * np.linalg.norm(Ab, axis=1)
        act_cos = float(np.mean(num / np.maximum(den, 1e-9)))
        jac = float(np.mean([len(a & b) / len(a | b) for a, b in zip(topk[i - 1], topk[i])]))
        wm = float((wins[i - 1] == wins[i]).mean())
        print(f"{f'{eps[i-1]}->{eps[i]}':>12} | {dW:>7.3f} | {act_cos:>7.3f} | {jac:>8.3f} | {wm:>9.3f}")


if __name__ == "__main__":
    main()
