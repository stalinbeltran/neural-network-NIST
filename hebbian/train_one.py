"""Entrena una red NUEVA con UNA SOLA imagen y graba, epoca a epoca, el conjunto de neuronas que
DISPARAN (activacion >= theta). Guarda un fotograma solo cuando ese conjunto CAMBIA respecto al
anterior (si varias epocas seguidas no cambia nada, no se duplica). Para verlo en webapp_evolution.py.

Uso:
    python hebbian/train_one.py --lr 0.02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena con 1 imagen y graba los cambios del disparo")
    ap.add_argument("--source", type=Path, default=LINES_NPZ)
    ap.add_argument("--img-index", type=int, default=-1, help="indice de la imagen (-1 = aleatoria)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--theta", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=6000)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("experiments/one_evo/frames.npz"))
    args = ap.parse_args()

    imgs = np.load(args.source)["images"].reshape(-1, 784).astype(np.float32) / 255.0
    idx = args.img_index if args.img_index >= 0 else int(np.random.default_rng(args.seed).integers(len(imgs)))
    x0 = imgs[idx]
    layer = CompetitiveLayer(784, 2500, reinforce_gain=args.reinforce_gain,
                             inhib_on=True, inhib_gain=args.inhib_gain, seed=args.seed)
    xu = layer._normalize_vec(x0)
    rng = np.random.default_rng(args.seed + 1)
    X = x0[None, :]
    thr = args.theta

    acts, eps = [], []
    a = layer.W @ xu
    last = a >= thr
    acts.append(a.astype(np.float16)); eps.append(0)
    for ep in range(1, args.max_epochs + 1):
        layer.learn_epoch(X, args.lr, rng)                 # 1 imagen
        a = layer.W @ xu
        fired = a >= thr
        if not np.array_equal(fired, last):                # graba solo si cambia el conjunto de disparo
            acts.append(a.astype(np.float16)); eps.append(ep); last = fired
            if len(acts) >= args.max_frames:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, acts=np.stack(acts), epochs=np.array(eps),
                        input=(x0 * 255).astype(np.uint8), theta=thr, img_index=idx)
    print(f"imagen #{idx}  |  {len(acts)} fotogramas (cambios del disparo) hasta ep{eps[-1]}  ->  {args.out}")


if __name__ == "__main__":
    main()
