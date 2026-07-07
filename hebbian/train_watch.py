"""Entrena (o reanuda) una red y REGISTRA los parametros de estabilidad EN CADA EPOCA, para ver el
progreso. Pensado para tramos acumulables: se guarda un model.npz reanudable y se ANEXA al CSV.

Por cada epoca calcula, comparando con la epoca anterior sobre el set de entrenamiento:
  - dead      : neuronas que nunca han ganado.
  - fired     : media de neuronas que disparan por entrada (theta).
  - act       : activacion media del ganador (nitidez).
  - dW_rel    : cuanto cambiaron los pesos respecto a la epoca previa (0=iguales).
  - act_cos   : cuanto se conserva el patron de respuesta por entrada (1=igual).
  - win_match : fraccion de entradas cuya neurona ganadora es la MISMA que la epoca previa.

Uso:
    python hebbian/train_watch.py --dataset data/processed/lines_hebbian/lines_20.npz \
        --model experiments/watch20/model.npz --csv experiments/watch20/metrics.csv \
        --epochs 500 --lr 0.02
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from competitive_net import CompetitiveLayer


def load_X(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"].reshape(-1, 784).astype(np.float32)
    if imgs.max() > 1.0:
        imgs /= 255.0
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena registrando parametros de estabilidad por epoca")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="model.npz (reanuda si existe, y se guarda)")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=500, help="epocas a entrenar en este tramo")
    ap.add_argument("--minutes", type=float, default=0.0, help="corta el tramo tras estos minutos (0=sin limite)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--milestone-dir", type=Path, default=None)
    ap.add_argument("--milestone-every", type=int, default=1000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    X = load_X(args.dataset)
    if args.model.exists():
        layer = CompetitiveLayer.load(args.model)
    else:
        args.model.parent.mkdir(parents=True, exist_ok=True)
        layer = CompetitiveLayer(784, 2500, reinforce_gain=args.reinforce_gain,
                                 inhib_on=True, inhib_gain=args.inhib_gain, seed=args.seed)
    Xn = layer._normalize_rows(X)
    thr = args.threshold
    rng = np.random.default_rng(args.seed + layer.epochs_trained + 1)   # continua distinto cada tramo

    fields = ["epoch", "lr", "dead", "fired", "act", "dW_rel", "act_cos", "win_match"]
    new_csv = not args.csv.exists()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(args.csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    if new_csv:
        w.writeheader()

    A_prev = Xn @ layer.W.T
    win_prev = A_prev.argmax(1)
    W_prev = layer.W.copy()

    t0 = time.time()
    done = 0
    for _ in range(args.epochs):
        layer.learn_epoch(X, args.lr, rng)
        A = Xn @ layer.W.T
        win = A.argmax(1)
        num = (A_prev * A).sum(1)
        den = np.linalg.norm(A_prev, axis=1) * np.linalg.norm(A, axis=1)
        row = {
            "epoch": layer.epochs_trained,
            "lr": args.lr,
            "dead": int(layer.n_out - np.count_nonzero(layer.win_count)),
            "fired": round(float((A >= thr).sum(1).mean()), 1),
            "act": round(float(A.max(1).mean()), 4),
            "dW_rel": round(float(np.linalg.norm(layer.W - W_prev) / np.linalg.norm(W_prev)), 4),
            "act_cos": round(float(np.mean(num / np.maximum(den, 1e-9))), 4),
            "win_match": round(float((win_prev == win).mean()), 3),
        }
        w.writerow(row)
        A_prev, win_prev, W_prev = A, win, layer.W.copy()
        done += 1
        if args.milestone_dir and layer.epochs_trained % args.milestone_every == 0:
            args.milestone_dir.mkdir(parents=True, exist_ok=True)
            layer.save(args.milestone_dir / f"model_ep{layer.epochs_trained:06d}.npz")
        if args.minutes and (time.time() - t0) / 60.0 >= args.minutes:
            break
    f.close()
    layer.save(args.model)

    last = row
    print(f"tramo: +{done} epocas -> total {layer.epochs_trained}  ({time.time()-t0:.0f}s)")
    print(f"ultima epoca: dead={last['dead']} fired={last['fired']} act={last['act']} "
          f"dW_rel={last['dW_rel']} act_cos={last['act_cos']} win_match={last['win_match']}")
    print(f"NOW={datetime.now():%H:%M:%S}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np
            data = _np.genfromtxt(args.csv, delimiter=",", names=True)
            ep = data["epoch"]
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
            ax[0].plot(ep, data["dW_rel"], label="dW_rel (cambio pesos)")
            ax[0].plot(ep, data["act_cos"], label="act_cos (patron)")
            ax[0].plot(ep, data["win_match"], label="win_match (ganadora)")
            ax[0].set(title="Estabilidad vs epoca (1=estable)", xlabel="epoca", ylabel="fraccion", ylim=(0, 1))
            ax[0].legend(fontsize=8)
            ax[1].plot(ep, data["fired"], label="disparos/entrada")
            ax[1].plot(ep, data["act"] * 100, label="act ganador x100")
            ax[1].set(title="Disparos y nitidez vs epoca", xlabel="epoca")
            ax[1].legend(fontsize=8)
            fig.tight_layout()
            png = args.csv.with_suffix(".png")
            fig.savefig(png, dpi=110)
            plt.close(fig)
            print(f"grafica -> {png}")
        except Exception as e:
            print(f"(sin grafica: {e})")


if __name__ == "__main__":
    main()
