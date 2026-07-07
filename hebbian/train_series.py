"""Entrena una red NUEVA y guarda un snapshot cada `--snapshot-every` epocas hasta `--total`
(por defecto 5 snapshots: ep10/20/30/40/50). Pensado para estudiar la dinamica del entrenamiento
(estabilidad, cantidad de disparos, etc.) con analyze_series.py.

Uso:
    python hebbian/train_series.py                 # 50 epocas, snapshot cada 10, rgain=1, igain=1.5
    python hebbian/train_series.py --dataset data/processed/lines_hebbian/lines_posneg.npz
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
    ap = argparse.ArgumentParser(description="Entrena una red y guarda snapshots cada N epocas")
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ)
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--lr0", type=float, default=0.1, help="learning rate inicial")
    ap.add_argument("--lr-min", type=float, default=0.005, help="learning rate final (annealing)")
    ap.add_argument("--n-out", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None,
                    help="continua desde un model.npz (snapshots numerados por epoca acumulada)")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/series_r1_g15"))
    args = ap.parse_args()

    X = load_X(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        layer = CompetitiveLayer.load(args.resume)
        print(f"reanudando desde {args.resume} (epocas previas={layer.epochs_trained})")
    else:
        layer = CompetitiveLayer(X.shape[1], args.n_out, reinforce_gain=args.reinforce_gain,
                                 inhib_on=True, inhib_gain=args.inhib_gain, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"dataset {args.dataset} ({len(X)})  rgain={layer.reinforce_gain} igain={layer.inhib_gain} "
          f"lr={args.lr0}->{args.lr_min}  -> {args.out_dir}")

    rows = []
    for ep in range(1, args.total + 1):
        frac = (ep - 1) / max(args.total - 1, 1)
        lr = args.lr0 * (args.lr_min / args.lr0) ** frac      # decae exponencial lr0 -> lr_min
        m = layer.learn_epoch(X, lr, rng)                     # incrementa layer.epochs_trained
        cum = layer.epochs_trained                            # epoca acumulada (para nombrar snapshots)
        m["epoch"] = cum
        m["lr"] = lr
        rows.append({k: m[k] for k in ("epoch", "lr", "mean_winner_activation", "coverage",
                                       "unique_winners", "dead_units", "mean_fired")})
        tag = ""
        if ep % args.snapshot_every == 0:
            layer.save(args.out_dir / f"model_ep{cum:03d}.npz")
            tag = "  [snapshot]"
        print(f"ep {cum:03d} (run {ep:02d}/{args.total})  lr={lr:.3f} act={m['mean_winner_activation']:.3f} "
              f"cob={m['coverage']:.2f} ganad={m['unique_winners']} muertas={m['dead_units']} "
              f"disp/ent={m['mean_fired']:.1f}{tag}")

    with open(args.out_dir / "train_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nlisto. snapshots en {args.out_dir}\\model_ep*.npz ; metricas en train_metrics.csv")


if __name__ == "__main__":
    main()
