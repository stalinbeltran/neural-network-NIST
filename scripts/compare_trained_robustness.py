"""Compara la ROBUSTEZ de dos modelos ENTRENADOS: uno con datos limpios vs. uno con ruido.

Carga dos checkpoints (formato reanudable con `model_state`), reconstruye la CNN a partir de
los propios pesos y evalúa AMBOS sobre el TEST limpio y sobre todos los subsets ruidosos
disponibles (tipo × nivel, cacheados). Responde: ¿entrenar con ruido mejora la robustez, y
generaliza a tipos de ruido distintos del visto en entrenamiento?

Uso:
  python scripts/compare_trained_robustness.py
  python scripts/compare_trained_robustness.py --clean <ckpt> --noisy <ckpt>
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

CKPT_DIR = Path("experiments/_noise_compare")
OUT_PNG = Path("reports/trained_noise_robustness.png")
LEVELS = [1, 2, 3, 4, 5]


def build_from_checkpoint(path: str, input_shape, num_classes):
    """Reconstruye la CNN a partir de los pesos guardados (infiere hiperparámetros) y la carga."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model_state"]
    channels = [v.shape[0] for k, v in sd.items()
                if k.startswith("features") and k.endswith(".weight") and v.dim() == 4]
    kernel = next(v.shape[2] for k, v in sd.items()
                  if k.startswith("features") and k.endswith(".weight") and v.dim() == 4)
    batchnorm = any("running_mean" in k for k in sd)
    fc_hidden = sd["classifier.1.weight"].shape[0] if "classifier.1.weight" in sd else 0
    lin_idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("classifier") and k.endswith(".weight"))
    dropout = 0.5 if lin_idx and lin_idx[-1] == 4 else 0.0   # dropout presente -> 2º Linear en índice 4

    model = build_model("cnn", input_shape=input_shape, num_classes=num_classes,
                        channels=channels, kernel_size=kernel, fc_hidden=fc_hidden,
                        batchnorm=batchnorm, dropout=dropout)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", default=str(CKPT_DIR / "ckpt_clean.pt"))
    ap.add_argument("--noisy", default=str(CKPT_DIR / "ckpt_noisy.pt"))
    ap.add_argument("--out", default=str(OUT_PNG))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    bundle = load_mnist()
    input_shape, num_classes = bundle.input_shape, bundle.num_classes

    # TEST limpio como dataset
    xs, ys = zip(*[(x, int(y)) for x, y in bundle.test])
    clean_ds = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    models = {}
    for tag, path in [("limpio", args.clean), ("ruidoso", args.noisy)]:
        model, ckpt = build_from_checkpoint(path, input_shape, num_classes)
        trainer = Trainer(model, TrainConfig(device="cpu", batch_size=args.batch_size))
        meta = ckpt.get("meta", {})
        trained_on = ckpt.get("history", {})
        models[tag] = {"trainer": trainer, "ckpt": ckpt}

    def acc(trainer, dataset):
        return trainer.evaluate(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))[0]

    # entrenado con: ruido gaussiano nivel_3 (según el checkpoint noisy)
    n_ck = models["ruidoso"]["ckpt"]
    train_tipo = n_ck["history"].get("tipo") if isinstance(n_ck.get("history"), dict) else None
    train_tipo = train_tipo or "gaussiano"
    print(f"Modelo LIMPIO  entrenado en datos limpios")
    print(f"Modelo RUIDOSO entrenado con ruido: gaussiano nivel_3 (según checkpoint)\n")

    clean_acc = {t: acc(models[t]["trainer"], clean_ds) for t in models}
    print(f"TEST LIMPIO   -> limpio={clean_acc['limpio']:.4f}  ruidoso={clean_acc['ruidoso']:.4f}  "
          f"(delta={clean_acc['ruidoso']-clean_acc['limpio']:+.4f})\n")

    levels_cfg = load_levels()
    types = list(levels_cfg["types"])
    results = {"clean_accuracy": clean_acc, "noise": {}}

    header = "tipo".ljust(22) + "  " + "".join(f"n{n}".rjust(8) for n in LEVELS) + "   (limpio / ruidoso)"
    print(header + "\n" + "-" * 78)
    for tipo in types:
        row = {"limpio": {}, "ruidoso": {}}
        line = tipo.ljust(22) + "  "
        for n in LEVELS:
            ds = noisy_dataset(tipo, f"nivel_{n}", "test")
            a_c = acc(models["limpio"]["trainer"], ds)
            a_n = acc(models["ruidoso"]["trainer"], ds)
            row["limpio"][f"nivel_{n}"] = a_c
            row["ruidoso"][f"nivel_{n}"] = a_n
            line += f"{a_n - a_c:+8.3f}"
        results["noise"][tipo] = row
        print(line + "   (delta ruidoso-limpio)")

    # --- figura: subplot por tipo, curva limpio vs ruidoso ---
    ncols = 4
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, tipo in enumerate(types):
        ax = axes[i // ncols][i % ncols]
        for tag, color in [("limpio", "tab:blue"), ("ruidoso", "tab:orange")]:
            ys_ = [results["noise"][tipo][tag][f"nivel_{n}"] for n in LEVELS]
            ax.plot(LEVELS, ys_, marker="o", label=f"entren. {tag}", color=color)
        ax.set_title(tipo, fontsize=10)
        ax.set_xticks(LEVELS)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(types), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Robustez: CNN entrenada en LIMPIO vs. entrenada con RUIDO (gaussiano n3)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115)
    plt.close(fig)

    (CKPT_DIR / "robustness_compare.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOK: figura en {args.out} | resultados en {CKPT_DIR / 'robustness_compare.json'}")


if __name__ == "__main__":
    main()
