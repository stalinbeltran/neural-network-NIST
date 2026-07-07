"""Barrido de rango de la GANANCIA INHIBIDORA: para varios valores de inhib_gain, parte del MISMO
checkpoint y entrena unas pocas epocas, midiendo como cambia la media de neuronas disparadas por
entrada (mean_fired). Sirve para encontrar con que ganancia se recorta el exceso de disparos.

La ganancia inhibidora es independiente del lr (reduccion = inhib_gain * exceso), asi que este
barrido aisla su efecto. Todos los runs usan la misma semilla de shuffle -> las diferencias son solo
por la ganancia.

Uso:
    python hebbian/sweep_inhib_gain.py
    python hebbian/sweep_inhib_gain.py --start experiments/inhib_series/model_ep020.npz \
        --gains 0 0.5 1 2 4 8 16 --epochs 5 --lr 0.1
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


def mean_fired(layer: CompetitiveLayer, Xn: np.ndarray) -> float:
    return float((Xn @ layer.W.T >= layer.fire_threshold).sum(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido de rango de la ganancia inhibidora")
    ap.add_argument("--start", type=Path, default=Path("experiments/inhib_series/model_ep020.npz"))
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ)
    ap.add_argument("--gains", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8, 16])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("experiments/inhib_series/gain_sweep"))
    args = ap.parse_args()

    X = load_X(args.dataset)
    base = CompetitiveLayer.load(args.start)
    Xn = base._normalize_rows(X)
    theta = base.fire_threshold
    start_fired = mean_fired(base, Xn)
    print(f"checkpoint de partida: {args.start}  (epochs={base.epochs_trained}, theta={theta})")
    print(f"mean_fired inicial = {start_fired:.1f} neuronas/entrada\n")
    print(f"{'gain':>6} | {'inicio':>7} | " + " | ".join(f"ep{e+1:>2}" for e in range(args.epochs)) + " | delta")

    rows = []
    trajectories = {}
    for gain in args.gains:
        layer = CompetitiveLayer.load(args.start)
        layer.configure_inhibition(spacing=layer.inhib_spacing, radius=layer.inhib_radius,
                                   metric=layer.inhib_metric, fire_threshold=layer.fire_threshold,
                                   K=layer.inhib_K, gain=gain, mode=layer.inhib_mode)
        rng = np.random.default_rng(args.seed)
        traj = [start_fired]
        for _ in range(args.epochs):
            m = layer.learn_epoch(X, args.lr, rng)
            traj.append(m["mean_fired"])
        trajectories[gain] = traj
        delta = traj[-1] - start_fired
        rows.append({"gain": gain, "start": start_fired, "final": traj[-1], "delta": delta,
                     "dead_units_final": int(layer.n_out - np.count_nonzero(layer.win_count))})
        mid = " | ".join(f"{v:5.0f}" for v in traj[1:])
        print(f"{gain:>6} | {start_fired:>7.1f} | {mid} | {delta:+6.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gain", "start", "final", "delta", "dead_units_final"])
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for gain, traj in trajectories.items():
            ax[0].plot(range(len(traj)), traj, "-o", ms=3, label=f"gain={gain}")
        ax[0].axhline(start_fired, ls="--", c="gray", lw=1)
        ax[0].set(title="Disparos/entrada vs epoca", xlabel="epoca", ylabel="mean_fired")
        ax[0].legend(fontsize=8)
        g = [r["gain"] for r in rows]
        ax[1].plot(g, [r["final"] for r in rows], "-o")
        ax[1].axhline(start_fired, ls="--", c="gray", lw=1, label="inicio")
        ax[1].set(title="mean_fired final vs ganancia", xlabel="inhib_gain", ylabel="mean_fired final")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.out.with_suffix(".png"), dpi=110)
        plt.close(fig)
        print(f"\ngrafica -> {args.out.with_suffix('.png')}")
    except Exception as e:
        print(f"(sin grafica: {e})")
    print(f"csv     -> {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
