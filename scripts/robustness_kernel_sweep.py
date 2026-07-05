"""Evalúa la ROBUSTEZ de las 3 CNN del kernel-sweep (k=3/5/7) sobre el TEST con ruidos variados.

Checkpoints reanudables del Trainer (dict con `model_state`) en experiments/_kernel_sweep/.
Arquitectura común: channels=[16,32,64], dropout=0.2, fc_hidden=128, sin batchnorm; solo cambia
kernel_size. Entrenadas en datos LIMPIOS. Se reconstruye cada modelo (mismo diseño, distinto
kernel), se carga y se evalúa sobre el TEST limpio y todos los subsets ruidosos (tipo × nivel).

Uso:  python scripts/robustness_kernel_sweep.py
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

CKPT_DIR = Path("experiments/_kernel_sweep")
OUT_PNG = Path("reports/robustness_kernel_sweep.png")
LEVELS = [1, 2, 3, 4, 5]
CKPTS = {"k3": "ckpt_k3.pt", "k5": "ckpt_k5.pt", "k7": "ckpt_k7.pt"}


def build_from_checkpoint(path, input_shape, num_classes):
    """Reconstruye la CNN con el MISMO diseño que se entrenó (kernel inferido de los pesos)."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model_state"]
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
    return model, kernel, ckpt.get("epochs_done"), model.count_params()["params_total"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR),
                    help="Carpeta con ckpt_k3/k5/k7.pt (usa el snapshot congelado, no los que siguen entrenando).")
    ap.add_argument("--out", default=str(OUT_PNG))
    args = ap.parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_png = Path(args.out)

    bundle = load_mnist()
    input_shape, num_classes = bundle.input_shape, bundle.num_classes
    xs, ys = zip(*[(x, int(y)) for x, y in bundle.test])
    clean_ds = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    trainers, info = {}, {}
    for tag, fname in CKPTS.items():
        model, kernel, epochs, params = build_from_checkpoint(ckpt_dir / fname, input_shape, num_classes)
        trainers[tag] = Trainer(model, TrainConfig(device="cpu", batch_size=256))
        info[tag] = {"kernel": kernel, "epochs_done": epochs, "params": params}

    def acc(tag, ds):
        return trainers[tag].evaluate(DataLoader(ds, batch_size=256, shuffle=False))[0]

    results = {"info": info, "clean_accuracy": {}, "noise": {}}
    print("Modelos:")
    for tag in CKPTS:
        clean = acc(tag, clean_ds)
        results["clean_accuracy"][tag] = clean
        print(f"  {tag} | kernel={info[tag]['kernel']} | params={info[tag]['params']:,} | "
              f"epochs_done={info[tag]['epochs_done']} | TEST limpio={clean:.4f}")
    print()

    levels_cfg = load_levels()
    types = list(levels_cfg["types"])
    tags = list(CKPTS)

    header = "tipo".ljust(22) + "".join(f"n{n}".rjust(8) for n in LEVELS)
    for tag in tags:
        print(f"[{tag}]  " + header)
    print("-" * (22 + 8 * len(LEVELS)))
    for tipo in types:
        results["noise"][tipo] = {}
        for tag in tags:
            row = {}
            line = f"[{tag}] " + tipo.ljust(21)
            for n in LEVELS:
                a = acc(tag, noisy_dataset(tipo, f"nivel_{n}", "test"))
                row[f"nivel_{n}"] = a
                line += f"{a:8.3f}"
            results["noise"][tipo][tag] = row
            print(line)
        print()

    # --- figura overlay: subplot por tipo, una curva por kernel ---
    ncols = 4
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, tipo in enumerate(types):
        ax = axes[i // ncols][i % ncols]
        for tag in tags:
            ys_ = [results["noise"][tipo][tag][f"nivel_{n}"] for n in LEVELS]
            ax.plot(LEVELS, ys_, marker="o", label=f"{tag} (k={info[tag]['kernel']})")
        ax.set_title(tipo, fontsize=10)
        ax.set_xticks(LEVELS)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(types), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Robustez al ruido — kernel sweep (k=3/5/7), CNN [16,32,64]", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=115)
    plt.close(fig)

    (ckpt_dir / "robustness_kernel.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: figura en {out_png} | resultados en {ckpt_dir / 'robustness_kernel.json'}")


if __name__ == "__main__":
    main()
