"""Encuentra automáticamente un learning rate inicial para una config (LR range test).

Uso:
  python scripts/lr_find.py --config configs/models/cnn_shapes_tiny.yaml
  python scripts/lr_find.py --config configs/models/cnn_shapes_tiny.yaml --plot experiments/lr_curve.png

Imprime el LR sugerido (para copiar a `train.lr`). No entrena el modelo: solo lo sondea.
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from nnist.data import load_dataset
from nnist.experiments import load_config
from nnist.models import build_model
from nnist.training import find_lr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-iter", type=int, default=100, help="nº de batches del sondeo")
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--lr-max", type=float, default=1.0)
    ap.add_argument("--plot", default=None, help="ruta PNG opcional para la curva loss vs lr")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds_kwargs = {k: v for k, v in cfg.dataset.items() if k != "name"}
    bundle = load_dataset(cfg.dataset["name"], **ds_kwargs)
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    model = build_model(cfg.model["name"], input_shape=bundle.input_shape,
                        num_classes=bundle.num_classes, **model_kwargs)

    train = dict(cfg.train)
    loader = DataLoader(bundle.train, batch_size=train.get("batch_size", 32), shuffle=True)
    res = find_lr(model, loader, optimizer=train.get("optimizer", "adam"),
                  weight_decay=train.get("weight_decay", 0.0), device=train.get("device", "cpu"),
                  lr_min=args.lr_min, lr_max=args.lr_max, num_iter=args.num_iter)

    print(f"LR sugerido: {res['suggested_lr']:.4g}   (lr actual en la config: {train.get('lr')})")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(res["lrs"], res["losses"])
            plt.axvline(res["suggested_lr"], color="r", ls="--", label=f"sugerido={res['suggested_lr']:.4g}")
            plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("loss (suavizada)")
            plt.legend(); plt.tight_layout(); plt.savefig(args.plot)
            print(f"Curva guardada en {args.plot}")
        except ImportError:
            print("matplotlib no disponible: instálalo para --plot.")


if __name__ == "__main__":
    main()
