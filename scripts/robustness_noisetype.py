"""Robustez del SWEEP de tipo de ruido: evalúa CADA modelo entrenado (uno por tipo de ruido de
entrenamiento + baseline limpio) sobre datos LIMPIOS y sobre TODOS los subsets ruidosos
(tipo × nivel). No reentrena: solo inferencia.

Responde: ¿entrenar con un tipo de ruido hace al modelo más robusto al ruido en general (o solo
a ese tipo)? Carga los checkpoints del Trainer (ckpt_<label>.pt) de una carpeta (por defecto la de
continuación), arquitectura fija CNN [16,32,64]+dropout.

Salidas (en --out-dir y --plot-prefix):
  - results.json / robustness.csv : matriz completa modelo × condición.
  - <prefix>_heatmap.png : accuracy media (sobre niveles) por modelo × tipo de ruido de test.
  - <prefix>_overall.png : robustez global (accuracy media sobre TODAS las condiciones ruidosas) por modelo.

Uso:
  python scripts/robustness_noisetype.py --ckpt-dir experiments/_noisetype_sweep_cont
"""
from __future__ import annotations

import argparse
import csv
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

BASELINE = "limpio"
CHANNELS = [16, 32, 64]


def _clean_test_dataset():
    bundle = load_mnist()
    xs, ys = [], []
    for x, y in bundle.test:
        xs.append(x)
        ys.append(int(y))
    return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))


def _load_model(ckpt_path: Path, dropout: float) -> Trainer:
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=10,
                        channels=CHANNELS, dropout=dropout)
    t = Trainer(model, TrainConfig(epochs=1, device="cpu"))
    t.resume_from(ckpt_path)          # arquitectura fija -> carga estricta
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="experiments/_noisetype_sweep_cont")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--types", nargs="+", default=None, help="tipos de ruido de test (por defecto todos)")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default=None, help="por defecto <ckpt-dir>/robustness")
    ap.add_argument("--plot-prefix", default="reports/robustness_noisetype")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    levels_cfg = load_levels()
    test_types = args.types or list(levels_cfg["types"])
    # modelos = baseline + un modelo por tipo (los que existan como checkpoint)
    model_labels = [BASELINE] + list(levels_cfg["types"])
    model_labels = [lb for lb in model_labels if (ckpt_dir / f"ckpt_{lb}.pt").exists()]
    print(f"Modelos: {len(model_labels)} | tipos test: {len(test_types)} | niveles: {args.levels}", flush=True)

    # carga los modelos (ligeros: 98k params c/u)
    trainers = {lb: _load_model(ckpt_dir / f"ckpt_{lb}.pt", args.dropout) for lb in model_labels}

    def acc_all(dataset) -> dict:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return {lb: trainers[lb].evaluate(loader)[0] for lb in model_labels}

    # condición LIMPIA (columna baseline)
    print("Evaluando LIMPIO...", flush=True)
    clean_acc = acc_all(_clean_test_dataset())

    # matriz modelo -> {tipo -> {nivel -> acc}}
    matrix = {lb: {"clean": clean_acc[lb], "noise": {}} for lb in model_labels}
    for tipo in test_types:
        for n in args.levels:
            nivel = f"nivel_{n}"
            accs = acc_all(noisy_dataset(tipo, nivel, "test"))
            for lb in model_labels:
                matrix[lb]["noise"].setdefault(tipo, {})[nivel] = accs[lb]
            print(f"  {tipo} {nivel}: " + " ".join(f"{lb[:4]}={accs[lb]:.3f}" for lb in model_labels[:4])
                  + " ...", flush=True)

    # persistencia
    (out_dir / "results.json").write_text(
        json.dumps({"ckpt_dir": str(ckpt_dir), "levels": args.levels, "matrix": matrix},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo_entrenado_con", "test_tipo", "test_nivel", "accuracy"])
        for lb in model_labels:
            w.writerow([lb, "clean", "-", f"{matrix[lb]['clean']:.4f}"])
            for tipo in test_types:
                for n in args.levels:
                    nivel = f"nivel_{n}"
                    w.writerow([lb, tipo, nivel, f"{matrix[lb]['noise'][tipo][nivel]:.4f}"])

    _plot_heatmap(matrix, model_labels, test_types, args.levels, f"{args.plot_prefix}_heatmap.png")
    _plot_overall(matrix, model_labels, test_types, args.levels, f"{args.plot_prefix}_overall.png")
    print(f"\nOK: {out_dir} (results.json, robustness.csv) + gráficas en {args.plot_prefix}_*.png", flush=True)


def _mean_over_levels(matrix, lb, tipo, levels):
    return float(np.mean([matrix[lb]["noise"][tipo][f"nivel_{n}"] for n in levels]))


def _plot_heatmap(matrix, model_labels, test_types, levels, out):
    """Filas = modelo (entrenado con tipo X), columnas = tipo de ruido de TEST; celda = acc media sobre niveles."""
    M = np.array([[_mean_over_levels(matrix, lb, tipo, levels) for tipo in test_types]
                  for lb in model_labels])
    fig, ax = plt.subplots(figsize=(1.0 * len(test_types) + 3, 0.55 * len(model_labels) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=max(0.0, M.min() - 0.02), vmax=1.0)
    ax.set_xticks(range(len(test_types))); ax.set_xticklabels(test_types, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_labels))); ax.set_yticklabels([f"train:{lb}" for lb in model_labels], fontsize=8)
    for i in range(len(model_labels)):
        for j in range(len(test_types)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.7 else "black")
    ax.set_title("Robustez: accuracy media (sobre niveles) — modelo (train-noise) × tipo de ruido de test")
    fig.colorbar(im, ax=ax, label="accuracy media")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Gráfica: {out}", flush=True)


def _plot_overall(matrix, model_labels, test_types, levels, out):
    """Barras: robustez GLOBAL = accuracy media sobre TODAS las condiciones ruidosas, por modelo."""
    overall = {}
    for lb in model_labels:
        vals = [matrix[lb]["noise"][tipo][f"nivel_{n}"] for tipo in test_types for n in levels]
        overall[lb] = float(np.mean(vals))
    items = sorted(overall.items(), key=lambda kv: kv[1])
    ys = [lb for lb, _ in items]; xs = [v for _, v in items]
    base = overall.get(BASELINE, 0.0)
    colors = ["tab:green" if v > base else ("gray" if lb == BASELINE else "tab:red")
              for (lb, v) in items]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ys, xs, color=colors)
    ax.axvline(base, color="black", ls="--", label=f"baseline (train limpio) = {base:.3f}")
    ax.set_xlabel("accuracy media sobre TODO el ruido (tipos × niveles)")
    ax.set_title("Robustez global por tipo de ruido de ENTRENAMIENTO\n"
                 "verde = más robusto que entrenar en limpio")
    ax.legend(loc="lower right"); ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Gráfica: {out}", flush=True)


if __name__ == "__main__":
    main()
