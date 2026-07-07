"""Analisis de estabilidad/OLVIDO del entrenamiento continual por rondas.

Construye la matriz retention[s, r] = activacion media del ganador del snapshot s (tras la ronda s)
sobre el SET de la ronda r. Interpretacion:
  - diagonal (s==r): rendimiento justo despues de entrenar ese set.
  - debajo de la diagonal (s>r): cuanto RETIENE un set antiguo tras aprender sets nuevos (olvido).
  - encima (s<r): respuesta a un set aun NO visto.
Tambien resume, por ronda, la retencion final vs recien-entrenado.

Uso:
    python hebbian/analyze_rounds.py --dir experiments/rounds_r1_g15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ


def load_pos(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"].reshape(-1, 784).astype(np.float32)
    if imgs.max() > 1.0:
        imgs /= 255.0
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Analisis de retencion/olvido por rondas")
    ap.add_argument("--dir", type=Path, default=Path("experiments/rounds_r1_g15"))
    ap.add_argument("--source", type=Path, default=LINES_NPZ)
    ap.add_argument("--threshold", type=float, default=0.40)
    args = ap.parse_args()

    paths = sorted(args.dir.glob("model_ep*.npz"))
    layers = [CompetitiveLayer.load(p) for p in paths]
    eps = [L.epochs_trained for L in layers]
    groups = np.load(args.dir / "rounds.npz")["groups"]      # (n_rounds, group_size)
    P = load_pos(args.source)
    nr = len(groups)
    thr = args.threshold

    # datos (normalizados) de cada ronda: 10 pos + 10 neg
    round_Xn = []
    for r in range(nr):
        pos = P[groups[r]]
        data = np.concatenate([pos, 1.0 - pos], axis=0)
        round_Xn.append(layers[0]._normalize_rows(data))

    act = np.zeros((len(layers), nr))
    fire = np.zeros((len(layers), nr))
    for s, L in enumerate(layers):
        for r in range(nr):
            A = round_Xn[r] @ L.W.T
            act[s, r] = A.max(1).mean()
            fire[s, r] = (A >= thr).sum(1).mean()

    print(f"snapshots (rondas) epocas={eps}   theta={thr}")
    print("\nretention[snapshot, set_ronda] = act. media del ganador (filas=tras ronda, cols=set):")
    print("      " + " ".join(f"S{r+1:>4}" for r in range(nr)))
    for s in range(len(layers)):
        print(f"tras{s+1:>2} " + " ".join(f"{act[s, r]:5.2f}" for r in range(nr)))

    print("\nRETENCION por ronda (recien-entrenado -> final):")
    print(f"{'ronda':>6} | {'recien':>7} | {'final':>7} | {'retiene%':>8}")
    for r in range(nr):
        just = act[r, r]
        fin = act[-1, r]
        print(f"{r+1:>6} | {just:>7.3f} | {fin:>7.3f} | {100*fin/max(just,1e-6):>7.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        im = ax[0].imshow(act, cmap="viridis", aspect="auto", vmin=act.min(), vmax=act.max())
        ax[0].set(title="Retencion: act. ganador (fila=tras ronda, col=set de ronda)",
                  xlabel="set de ronda", ylabel="snapshot tras ronda")
        ax[0].set_xticks(range(nr)); ax[0].set_xticklabels([f"S{r+1}" for r in range(nr)])
        ax[0].set_yticks(range(len(layers))); ax[0].set_yticklabels([f"{r+1}" for r in range(len(layers))])
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].plot(range(1, nr + 1), [act[r, r] for r in range(nr)], "-o", label="recien entrenado (diagonal)")
        ax[1].plot(range(1, nr + 1), [act[-1, r] for r in range(nr)], "-o", label="tras las 10 rondas (final)")
        ax[1].set(title="Olvido: respuesta a cada set, recien vs final", xlabel="set de ronda",
                  ylabel="act. media del ganador")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.dir / "rounds_retention.png", dpi=110)
        plt.close(fig)
        print(f"\ngrafica -> {args.dir / 'rounds_retention.png'}")
    except Exception as e:
        print(f"(sin grafica: {e})")


if __name__ == "__main__":
    main()
