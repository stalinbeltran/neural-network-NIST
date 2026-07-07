"""Evalúa el checkpoint de `train_inverted_noisy_mix` contra TODOS los tipos de ruido.

El modelo se entrenó sobre invertido+gaussiano_n3 y normal+gaussiano_n3 (dos polaridades, un
único tipo/nivel de ruido). Aquí medimos su robustez sobre TODO el catálogo de ruido de
configs/noise/levels.yaml (12 tipos × 5 niveles), en las DOS polaridades:

  - NORMAL   (fondo oscuro, trazo claro):  nnist.data.noisy_dataset
  - INVERTIDO(fondo claro, trazo oscuro):  nnist.data.inverted_noisy_dataset

Baseline: test limpio en ambas polaridades (sin ruido). gaussiano_n3 es in-distribution
(se entrenó con él); el resto de tipos son OUT-of-distribution -> mide generalización.

Salida en experiments/_inverted_noisy_mix/robustness/: results.json, robustness.csv, robustness.png

  python scripts/eval_inverted_noisy_mix.py
  python scripts/eval_inverted_noisy_mix.py --types gaussiano sal_pimienta --levels 1 3 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnist.data import (inverted_dataset, inverted_noisy_dataset, load_levels,
                        load_mnist, noisy_dataset)
from nnist.models import build_model
from nnist.training import TrainConfig, Trainer

CKPT = Path("experiments/_inverted_noisy_mix/ckpt_inverted_noisy_mix.pt")


def _clean_test(dataset_fn) -> TensorDataset:
    """TEST limpio (sin ruido) como TensorDataset, en la polaridad dada."""
    xs, ys = [], []
    for x, y in dataset_fn:
        xs.append(x if x.dim() == 3 else x)
        ys.append(int(y))
    return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--channels", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--types", nargs="+", default=None, help="Tipos de ruido (por defecto: todos).")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 1. reconstruir el modelo y cargar los pesos del checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=10,
                        channels=args.channels, dropout=args.dropout)
    model.load_state_dict(ckpt["model_state"])
    params = model.count_params()["params_total"]
    epochs_done = ckpt.get("epochs_done", "?")

    trainer = Trainer(model, TrainConfig(device="cpu", batch_size=args.batch_size))

    def acc_of(dataset) -> float:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return trainer.evaluate(loader)[0]

    # 2. baselines limpios en ambas polaridades
    bundle = load_mnist()
    clean_norm = acc_of(_clean_test(bundle.test))
    clean_inv = acc_of(inverted_dataset("test"))
    print(f"Checkpoint {args.ckpt} | época {epochs_done} | params={params:,}")
    print(f"LIMPIO  normal={clean_norm:.4f}  invertido={clean_inv:.4f}\n")

    # 3. barrido: cada tipo × nivel en las dos polaridades
    levels_cfg = load_levels()
    types = args.types or list(levels_cfg["types"])
    results = {
        "ckpt": args.ckpt, "epochs_done": epochs_done, "params_total": params,
        "clean": {"normal": clean_norm, "invertido": clean_inv},
        "noise": {},
    }

    lvl_cols = "".join(f"n{n}".rjust(8) for n in args.levels)
    header = "tipo".ljust(22) + "pol".ljust(5) + lvl_cols
    print(header)
    print("-" * len(header))
    for tipo in types:
        tdef = levels_cfg["types"][tipo]
        results["noise"][tipo] = {"param": tdef["param"], "normal": {}, "invertido": {}}
        for pol, ds_fn in (("normal", noisy_dataset),
                           ("invertido", lambda t, n, s: inverted_noisy_dataset(t, n, s))):
            base = clean_norm if pol == "normal" else clean_inv
            cells = (tipo if pol == "normal" else "").ljust(22) + pol[:4].ljust(5)
            for n in args.levels:
                nivel = f"nivel_{n}"
                a = acc_of(ds_fn(tipo, nivel, "test"))
                results["noise"][tipo][pol][nivel] = {
                    "accuracy": a, "value": tdef["levels"][nivel], "drop": base - a}
                cells += f"{a:8.4f}"
            print(cells, flush=True)

    # 4. persistir resultados
    out_dir = Path(args.ckpt).parent / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "param", "polaridad", "nivel", "value", "accuracy", "drop_vs_clean"])
        for tipo, td in results["noise"].items():
            for pol in ("normal", "invertido"):
                for nivel, cell in td[pol].items():
                    w.writerow([tipo, td["param"], pol, nivel, cell["value"],
                                f"{cell['accuracy']:.4f}", f"{cell['drop']:.4f}"])

    # 5. gráfica: dos paneles (normal / invertido), una curva por tipo de ruido
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, pol, base in ((axes[0], "normal", clean_norm), (axes[1], "invertido", clean_inv)):
        for tipo, td in results["noise"].items():
            ys = [td[pol][f"nivel_{n}"]["accuracy"] for n in args.levels]
            style = dict(marker="o", lw=2.2) if tipo == "gaussiano" else dict(marker=".", lw=1)
            ax.plot(args.levels, ys, label=tipo, **style)
        ax.axhline(base, ls="--", color="gray", label=f"limpio ({base:.3f})")
        ax.set_title(f"Polaridad {pol.upper()}")
        ax.set_xlabel("nivel de ruido (1 leve … 5 severo)")
        ax.set_xticks(args.levels); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy en test")
    axes[1].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(f"Robustez del modelo invertido+normal (gaussiano n3) — época {epochs_done}, "
                 f"{params:,} params\n(gaussiano = in-distribution, resto = OOD)")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness.png", dpi=120)
    plt.close(fig)

    print(f"\nOK: resultados en {out_dir} (results.json, robustness.csv, robustness.png)")


if __name__ == "__main__":
    main()
