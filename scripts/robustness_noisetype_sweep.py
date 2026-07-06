"""Generalización cruzada de ruido: cada CNN (entrenada con un tipo de ruido) vs. TODOS los sets.

Modelos del `_noisetype_sweep` (bitácora): una CNN entrenada en limpio + una por cada tipo de
ruido (nivel_2). Se evalúa cada modelo sobre el TEST limpio y todos los subsets ruidosos
(tipo × nivel). Responde: ¿entrenar con el ruido X ayuda contra el ruido Y?

Salidas en reports/:
  - robustness_noisetype_curves.png : subplot por tipo de TEST, una curva por modelo (acc vs nivel).
  - robustness_noisetype_heatmap.png: matriz entrenado-en (filas) × testeado-en (cols), media sobre niveles.

Uso:  python scripts/robustness_noisetype_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnist.data import load_levels, load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import TrainConfig, Trainer

CKPT_DIR = Path("experiments/_noisetype_sweep")
OUT_CURVES = Path("reports/robustness_noisetype_curves.png")
OUT_HEATMAP = Path("reports/robustness_noisetype_heatmap.png")
LEVELS = [1, 2, 3, 4, 5]


def build_from_checkpoint(path, input_shape, num_classes):
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
    return model, model.count_params()["params_total"]


def main() -> None:
    bundle = load_mnist()
    input_shape, num_classes = bundle.input_shape, bundle.num_classes
    xs, ys = zip(*[(x, int(y)) for x, y in bundle.test])
    clean_ds = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    types = list(load_levels()["types"])
    # orden de modelos: limpio primero, luego los tipos en orden del config
    order = ["limpio"] + types
    tags = [t for t in order if (CKPT_DIR / f"ckpt_{t}.pt").exists()]

    trainers, results = {}, {"models": {}}
    for tag in tags:
        model, params = build_from_checkpoint(CKPT_DIR / f"ckpt_{tag}.pt", input_shape, num_classes)
        trainers[tag] = Trainer(model, TrainConfig(device="cpu", batch_size=256))
        results["models"][tag] = {"params": params, "clean": None, "noise": {}}

    def acc(tag, ds):
        return trainers[tag].evaluate(DataLoader(ds, batch_size=256, shuffle=False))[0]

    print("Accuracy en TEST limpio por modelo (entrenado-en):")
    for tag in tags:
        results["models"][tag]["clean"] = acc(tag, clean_ds)
        print(f"  {tag:<22} {results['models'][tag]['clean']:.4f}")
    print("\nEvaluando cada modelo sobre todos los tipos×niveles...")

    for tag in tags:
        for tipo in types:
            results["models"][tag]["noise"][tipo] = {
                f"nivel_{n}": acc(tag, noisy_dataset(tipo, f"nivel_{n}", "test")) for n in LEVELS
            }
        print(f"  hecho: {tag}")

    CKPT_DIR.joinpath("robustness_noisetype.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Figura A: curvas, subplot por tipo de TEST, una línea por modelo ---
    cmap = plt.get_cmap("tab20")
    colors = {tag: cmap(i % 20) for i, tag in enumerate(tags)}
    ncols = 4
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, tipo in enumerate(types):
        ax = axes[i // ncols][i % ncols]
        for tag in tags:
            ys_ = [results["models"][tag]["noise"][tipo][f"nivel_{n}"] for n in LEVELS]
            lw = 2.4 if tag in ("limpio", tipo) else 1.0   # resalta limpio y el "entrenado en este ruido"
            ax.plot(LEVELS, ys_, marker="o", ms=3, lw=lw, color=colors[tag], label=tag)
        ax.set_title(f"test: {tipo}", fontsize=10)
        ax.set_xticks(LEVELS)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    # leyenda global en el hueco sobrante
    handles, labels = axes[0][0].get_legend_handles_labels()
    for j in range(len(types), nrows * ncols):
        ax = axes[j // ncols][j % ncols]
        ax.axis("off")
    axes[(len(types)) // ncols][(len(types)) % ncols].legend(
        handles, labels, fontsize=8, loc="center", title="entrenado en", ncol=1)
    fig.suptitle("Generalización cruzada de ruido — curva por modelo, subplot por tipo de test", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_CURVES.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CURVES, dpi=115)
    plt.close(fig)

    # --- Figura B: heatmap entrenado-en × testeado-en (media sobre niveles) ---
    M = np.zeros((len(tags), len(types)))
    for r, tag in enumerate(tags):
        for c, tipo in enumerate(types):
            M[r, c] = np.mean([results["models"][tag]["noise"][tipo][f"nivel_{n}"] for n in LEVELS])
    fig, ax = plt.subplots(figsize=(1.0 * len(types) + 3, 0.6 * len(tags) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=max(0.2, M.min()), vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(types)), types, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tags)), tags, fontsize=8)
    ax.set_xlabel("testeado en (tipo de ruido)")
    ax.set_ylabel("entrenado en")
    for r in range(len(tags)):
        for c in range(len(types)):
            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[r, c] < 0.7 else "black")
    fig.colorbar(im, ax=ax, label="accuracy media sobre niveles 1-5")
    ax.set_title("¿Entrenar con el ruido X ayuda contra el ruido Y? (media niveles 1-5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=120)
    plt.close(fig)

    print(f"\nOK: {OUT_CURVES} | {OUT_HEATMAP} | {CKPT_DIR/'robustness_noisetype.json'}")


if __name__ == "__main__":
    main()
