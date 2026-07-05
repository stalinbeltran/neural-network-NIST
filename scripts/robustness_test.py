"""Test de ROBUSTEZ: evalúa un modelo YA ENTRENADO (en limpio) sobre los subsets ruidosos.

Reconstruye el modelo desde el `config.yaml` de una corrida, carga su `model.pt`, y mide la
accuracy sobre el TEST limpio (baseline) y sobre cada subset ruidoso (tipo × nivel, generados
on-demand vía `nnist.data.noisy_dataset`). No reentrena nada: solo inferencia.

Salida en `experiments/<run>/robustness/`: results.json, robustness.csv y robustness.png
(accuracy vs nivel, una curva por tipo de ruido, con el baseline limpio de referencia).

Uso:
  python scripts/robustness_test.py --run experiments/cnn_arch_opt__channels=16-32-64__seed0_20260705_113814
  python scripts/robustness_test.py --run <dir> --types gaussiano sal_pimienta --levels 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from nnist.data import load_levels, load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import TrainConfig, Trainer

DEFAULT_RUN = "experiments/cnn_arch_opt__channels=16-32-64__seed0_20260705_113814"


def _clean_test_dataset():
    """TEST limpio como TensorDataset (mismas 10k imágenes, sin ruido)."""
    bundle = load_mnist()
    xs, ys = [], []
    for x, y in bundle.test:
        xs.append(x)
        ys.append(int(y))
    return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long)), bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=DEFAULT_RUN, help="Directorio de la corrida (config.yaml + model.pt).")
    ap.add_argument("--types", nargs="+", default=None, help="Tipos de ruido (por defecto: todos).")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Niveles a evaluar.")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))

    # 1. TEST limpio + input_shape/num_classes derivados de los datos
    clean_ds, bundle = _clean_test_dataset()
    input_shape, num_classes = bundle.input_shape, bundle.num_classes

    # 2. reconstruir el modelo desde su config y cargar el checkpoint entrenado
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], input_shape=input_shape,
                        num_classes=num_classes, **model_kwargs)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    params = model.count_params()["params_total"]

    trainer = Trainer(model, TrainConfig(device="cpu", batch_size=args.batch_size))

    def acc_of(dataset) -> float:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return trainer.evaluate(loader)[0]

    # 3. baseline limpio
    clean_acc = acc_of(clean_ds)
    print(f"Modelo {run_dir.name} | params={params:,} | accuracy LIMPIO = {clean_acc:.4f}\n")

    # 4. barrido de robustez: tipo × nivel (subsets ruidosos on-demand)
    levels_cfg = load_levels()
    types = args.types or list(levels_cfg["types"])
    results = {"run": run_dir.name, "params_total": params, "clean_accuracy": clean_acc, "noise": {}}

    header = "tipo".ljust(22) + "".join(f"n{n}".rjust(9) for n in args.levels)
    print(header)
    print("-" * len(header))
    for tipo in types:
        tdef = levels_cfg["types"][tipo]
        row = {}
        cells = tipo.ljust(22)
        for n in args.levels:
            nivel = f"nivel_{n}"
            a = acc_of(noisy_dataset(tipo, nivel, "test"))
            row[nivel] = {"accuracy": a, "value": tdef["levels"][nivel],
                          "drop": clean_acc - a}
            cells += f"{a:9.4f}"
        results["noise"][tipo] = {"param": tdef["param"], "levels": row}
        print(cells)

    # 5. persistir resultados + gráfica
    out_dir = run_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "param", "nivel", "value", "accuracy", "drop_vs_clean"])
        for tipo, td in results["noise"].items():
            for nivel, cell in td["levels"].items():
                w.writerow([tipo, td["param"], nivel, cell["value"],
                            f"{cell['accuracy']:.4f}", f"{cell['drop']:.4f}"])

    plt.figure(figsize=(9, 6))
    for tipo, td in results["noise"].items():
        xs = args.levels
        ys = [td["levels"][f"nivel_{n}"]["accuracy"] for n in xs]
        plt.plot(xs, ys, marker="o", label=tipo)
    plt.axhline(clean_acc, ls="--", color="gray", label=f"limpio ({clean_acc:.3f})")
    plt.xticks(args.levels)
    plt.xlabel("nivel de ruido (1 = leve ... 5 = severo)")
    plt.ylabel("accuracy en test")
    plt.title(f"Robustez al ruido — {cfg['model']['name']} ({params:,} params)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "robustness.png", dpi=120)
    plt.close()

    print(f"\nOK: resultados en {out_dir} (results.json, robustness.csv, robustness.png)")


if __name__ == "__main__":
    main()
