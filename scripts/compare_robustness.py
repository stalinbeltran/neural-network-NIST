"""Compara los perfiles de ROBUSTEZ de varias corridas (results.json de robustness_test.py).

Superpone, por tipo de ruido, la curva accuracy-vs-nivel de cada modelo, e imprime la tabla de
diferencias (delta) respecto al primero. Útil para responder "¿más parámetros = más robustez?".

Uso:
  python scripts/compare_robustness.py --runs experiments/<runA> experiments/<runB>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/robustness_compare.png")


def _load(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "robustness" / "results.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="Directorios de corrida a comparar.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data = [_load(r) for r in args.runs]
    labels = [f"{d['params_total']:,} params" for d in data]
    types = list(data[0]["noise"])
    levels = [1, 2, 3, 4, 5]

    # --- figura: un subplot por tipo, una curva por modelo ---
    ncols = 4
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, tipo in enumerate(types):
        ax = axes[i // ncols][i % ncols]
        for d, lab in zip(data, labels):
            ys = [d["noise"][tipo]["levels"][f"nivel_{n}"]["accuracy"] for n in levels]
            ax.plot(levels, ys, marker="o", label=lab)
        for d in data:
            ax.axhline(d["clean_accuracy"], ls=":", lw=0.8, alpha=0.5)
        ax.set_title(tipo, fontsize=10)
        ax.set_xticks(levels)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(types), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Robustez al ruido: comparación de modelos", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115)
    plt.close(fig)

    # --- tabla de deltas (modelo[1] - modelo[0]) en cada nivel ---
    base, other = data[0], data[-1]
    print(f"Clean: {base['params_total']:,} -> {base['clean_accuracy']:.4f} | "
          f"{other['params_total']:,} -> {other['clean_accuracy']:.4f}\n")
    print("delta (grande - pequeno), + = el grande es MAS robusto")
    header = "tipo".ljust(22) + "".join(f"n{n}".rjust(9) for n in levels)
    print(header + "\n" + "-" * len(header))
    for tipo in types:
        cells = tipo.ljust(22)
        for n in levels:
            a = base["noise"][tipo]["levels"][f"nivel_{n}"]["accuracy"]
            b = other["noise"][tipo]["levels"][f"nivel_{n}"]["accuracy"]
            cells += f"{b - a:+9.3f}"
        print(cells)
    print(f"\nOK: figura en {args.out}")


if __name__ == "__main__":
    main()
