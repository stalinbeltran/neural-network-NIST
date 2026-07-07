"""Barrido de la GANANCIA DE ACTIVACION (reinforce_gain): entrena una red NUEVA (pesos aleatorios)
por cada valor, con la inhibicion fija, y mide que representacion produce cada ganancia.

Metricas por red (tras N epocas):
  - dead_units  : neuronas que nunca ganan (menos = mejor uso del mapa).
  - coverage    : fraccion de neuronas que ganan algo en la ultima epoca.
  - unique_winners : nº de ganadoras distintas.
  - winner_act  : activacion media del ganador (mas alto = patrones mas nitidos).
  - mean_fired  : media de neuronas que disparan por entrada (esparsidad).

Un buen valor da POCAS neuronas muertas + buena cobertura + activacion del ganador razonable, sin
colapsar (unique_winners no se desploma). Guarda cada red entrenada para poder verla en la app.

Uso:
    python hebbian/sweep_reinforce_gain.py
    python hebbian/sweep_reinforce_gain.py --gains 0.25 0.5 1 2 4 8 --epochs 6 --inhib-gain 1.5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ


def load_X(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido de la ganancia de activacion (reinforce_gain)")
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ)
    ap.add_argument("--gains", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--n-out", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("experiments/reinforce_sweep/sweep"))
    ap.add_argument("--save-models", action="store_true", default=True,
                    help="guarda cada red entrenada (para verla luego en la app)")
    args = ap.parse_args()

    X = load_X(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {args.dataset} ({len(X)} entradas)  inhib_gain={args.inhib_gain}  lr={args.lr}  epochs={args.epochs}\n")
    print(f"{'r_gain':>7} | {'muertas':>7} | {'cobert':>6} | {'ganad':>5} | {'act_gan':>7} | {'disp/ent':>8}")
    print("-" * 58)

    rows = []
    for gain in args.gains:
        layer = CompetitiveLayer(X.shape[1], args.n_out, reinforce_gain=gain, inhib_on=True,
                                 inhib_gain=args.inhib_gain, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        m = {}
        for _ in range(args.epochs):
            m = layer.learn_epoch(X, args.lr, rng)
        row = {"reinforce_gain": gain, "dead_units": m["dead_units"], "coverage": round(m["coverage"], 3),
               "unique_winners": m["unique_winners"], "winner_act": round(m["mean_winner_activation"], 3),
               "mean_fired": round(m["mean_fired"], 1)}
        rows.append(row)
        print(f"{gain:>7} | {row['dead_units']:>7} | {row['coverage']:>6} | {row['unique_winners']:>5} | "
              f"{row['winner_act']:>7} | {row['mean_fired']:>8}")
        if args.save_models:
            layer.save(args.out.parent / f"model_rgain{gain:g}.npz")

    with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = [r["reinforce_gain"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(g, [r["dead_units"] for r in rows], "-o")
        ax[0].set(title="Neuronas muertas vs ganancia", xlabel="reinforce_gain", ylabel="muertas (menos=mejor)")
        ax[0].set_xscale("log", base=2)
        ax[1].plot(g, [r["unique_winners"] for r in rows], "-o")
        ax[1].set(title="Ganadoras unicas vs ganancia", xlabel="reinforce_gain", ylabel="ganadoras")
        ax[1].set_xscale("log", base=2)
        ax[2].plot(g, [r["winner_act"] for r in rows], "-o", label="act ganador")
        ax[2].set(title="Nitidez (act. del ganador) vs ganancia", xlabel="reinforce_gain", ylabel="cos sim")
        ax[2].set_xscale("log", base=2)
        fig.tight_layout()
        fig.savefig(args.out.with_suffix(".png"), dpi=110)
        plt.close(fig)
        print(f"\ngrafica -> {args.out.with_suffix('.png')}")
    except Exception as e:
        print(f"(sin grafica: {e})")
    print(f"csv     -> {args.out.with_suffix('.csv')}")
    if args.save_models:
        print(f"modelos -> {args.out.parent}\\model_rgain*.npz  (verlos en la app con --model)")


if __name__ == "__main__":
    main()
