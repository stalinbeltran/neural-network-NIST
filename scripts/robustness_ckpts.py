"""Robustez comparada de N checkpoints reananudables (formato Trainer con `model_state`).

Evalúa cada modelo sobre el TEST limpio y todos los subsets ruidosos (tipo × nivel) y superpone
las curvas (subplot por tipo, una línea por modelo). General y reutilizable: los modelos se pasan
como `etiqueta=ruta`.

Uso:
  python scripts/robustness_ckpts.py --ckpts limpio=exp/a.pt gaussiano=exp/b.pt --out reports/x.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnist.data import load_levels, load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import TrainConfig, Trainer

LEVELS = [1, 2, 3, 4, 5]


def build_from_checkpoint(path, input_shape, num_classes):
    sd = torch.load(path, map_location="cpu")["model_state"]
    channels = [v.shape[0] for k, v in sd.items()
                if k.startswith("features") and k.endswith(".weight") and v.dim() == 4]
    kernel = int(sd["features.0.weight"].shape[2])
    batchnorm = any("running_mean" in k for k in sd)
    fc_hidden = sd["classifier.1.weight"].shape[0]
    model = build_model("cnn", input_shape=input_shape, num_classes=num_classes,
                        channels=channels, kernel_size=kernel, fc_hidden=fc_hidden,
                        batchnorm=batchnorm, dropout=0.2)
    model.load_state_dict(sd)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True, help="pares etiqueta=ruta")
    ap.add_argument("--out", default="reports/robustness_ckpts.png")
    args = ap.parse_args()
    pairs = [c.split("=", 1) for c in args.ckpts]

    bundle = load_mnist()
    input_shape, num_classes = bundle.input_shape, bundle.num_classes
    xs, ys = zip(*[(x, int(y)) for x, y in bundle.test])
    clean_ds = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    trainers = {lab: Trainer(build_from_checkpoint(p, input_shape, num_classes),
                             TrainConfig(device="cpu", batch_size=256)) for lab, p in pairs}
    labels = [lab for lab, _ in pairs]

    def acc(lab, ds):
        return trainers[lab].evaluate(DataLoader(ds, batch_size=256, shuffle=False))[0]

    results = {"clean": {}, "noise": {}}
    print("TEST limpio:")
    for lab in labels:
        results["clean"][lab] = acc(lab, clean_ds)
        print(f"  {lab:<20} {results['clean'][lab]:.4f}")

    types = list(load_levels()["types"])
    for tipo in types:
        results["noise"][tipo] = {}
        for lab in labels:
            results["noise"][tipo][lab] = {f"nivel_{n}": acc(lab, noisy_dataset(tipo, f"nivel_{n}", "test"))
                                           for n in LEVELS}
        print(f"  evaluado: {tipo}")

    # figura overlay
    ncols = 4
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, tipo in enumerate(types):
        ax = axes[i // ncols][i % ncols]
        for lab in labels:
            ys_ = [results["noise"][tipo][lab][f"nivel_{n}"] for n in LEVELS]
            ax.plot(LEVELS, ys_, marker="o", ms=3, label=lab)
        ax.set_title(tipo, fontsize=10)
        ax.set_xticks(LEVELS); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(types), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Robustez comparada de checkpoints", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115); plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOK: {out} | {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
