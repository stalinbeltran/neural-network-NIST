"""Entrenamiento CONTINUAL por rondas. En cada ronda se eligen `group_size` positivas NUEVAS
(disjuntas entre rondas) y se forman sus negativas correspondientes -> 2*group_size imagenes; se
entrena `epochs_per_round` epocas con ese set y se pasa a otro set. Se repite `n_rounds` veces sobre
la MISMA red. Guarda un snapshot por ronda (model_ep<epoca acumulada>.npz) + rounds.npz (los indices
de cada ronda) + eval_set.npz (union de todos los sets, para el analisis de estabilidad).

Uso:
    python hebbian/train_rounds.py                 # 10 rondas x 5 epocas, 10 pos + 10 neg por ronda
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
    ap = argparse.ArgumentParser(description="Entrenamiento continual por rondas (pos + neg)")
    ap.add_argument("--source", type=Path, default=LINES_NPZ, help="set de positivas (se generan sus negativas)")
    ap.add_argument("--n-rounds", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=10, help="positivas por ronda (se anaden sus negativas)")
    ap.add_argument("--epochs-per-round", type=int, default=5)
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-out", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None, help="continua desde un model.npz")
    ap.add_argument("--groups", type=Path, default=None, help="rounds.npz con los MISMOS grupos a reutilizar")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/rounds_r1_g15"))
    args = ap.parse_args()

    P = load_pos(args.source)
    if args.groups:
        groups = np.load(args.groups)["groups"]                 # reutiliza los mismos sets
    else:
        rng = np.random.default_rng(args.seed)
        groups = rng.choice(len(P), size=args.n_rounds * args.group_size, replace=False)
        groups = groups.reshape(args.n_rounds, args.group_size)  # rondas disjuntas
    n_rounds = len(groups)
    if args.resume:
        layer = CompetitiveLayer.load(args.resume)
        print(f"reanudando desde {args.resume} (ep previas={layer.epochs_trained})")
    else:
        layer = CompetitiveLayer(784, args.n_out, reinforce_gain=args.reinforce_gain,
                                 inhib_on=True, inhib_gain=args.inhib_gain, seed=args.seed)
    rng_train = np.random.default_rng(args.seed + 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{n_rounds} rondas x {args.epochs_per_round} epocas ; {args.group_size} pos + {args.group_size} neg/ronda "
          f"; rgain={layer.reinforce_gain} igain={layer.inhib_gain} lr={args.lr}  -> {args.out_dir}")

    for r in range(n_rounds):
        idx = groups[r]
        pos = P[idx]
        data = np.concatenate([pos, 1.0 - pos], axis=0)          # 10 pos + sus 10 neg
        m = {}
        for _ in range(args.epochs_per_round):
            m = layer.learn_epoch(data, args.lr, rng_train)
        cum = layer.epochs_trained
        layer.save(args.out_dir / f"model_ep{cum:03d}.npz")
        print(f"ronda {r + 1:2d}/{n_rounds} (ep{cum:03d})  set={sorted(idx.tolist())}  "
              f"act={m['mean_winner_activation']:.3f} muertas={m['dead_units']} disp/ent={m['mean_fired']:.1f}")

    np.savez(args.out_dir / "rounds.npz", groups=groups)
    posE = P[groups.flatten()]
    evalimgs = (np.concatenate([posE, 1.0 - posE], axis=0) * 255).astype(np.uint8).reshape(-1, 28, 28)
    np.savez_compressed(args.out_dir / "eval_set.npz", images=evalimgs)
    print(f"\nlisto. snapshots model_ep*.npz ; rounds.npz ; eval_set.npz ({len(evalimgs)} img: mitad pos, mitad neg)")


if __name__ == "__main__":
    main()
