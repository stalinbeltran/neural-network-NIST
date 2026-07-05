"""Agrega la ROBUSTEZ por arquitectura promediando varias semillas (media ± std).

Toma varias corridas (results.json de robustness_test.py), las agrupa por nº de parámetros
(= misma arquitectura) y, para cada tipo×nivel, calcula media y desviación entre semillas.
Así la comparación "¿más parámetros = más robustez?" no depende de una única semilla.

Uso:
  python scripts/aggregate_robustness.py --runs expA_seed0 expA_seed1 expB_seed0 expB_seed1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/robustness_variance.png")
LEVELS = [1, 2, 3, 4, 5]


def _load(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "robustness" / "results.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    runs = [_load(r) for r in args.runs]

    # agrupar por arquitectura (nº de params). Orden estable por params ascendente.
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in runs:
        groups[r["params_total"]].append(r)
    arch_params = sorted(groups)
    types = list(runs[0]["noise"])

    def agg(params: int, tipo: str, n: int):
        vals = [g["noise"][tipo]["levels"][f"nivel_{n}"]["accuracy"] for g in groups[params]]
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    def clean_agg(params: int):
        vals = [g["clean_accuracy"] for g in groups[params]]
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    # --- resumen de semillas por grupo ---
    print("Arquitecturas (por nº de params):")
    for p in arch_params:
        cm, cs = clean_agg(p)
        print(f"  {p:,} params | {len(groups[p])} semillas | clean = {cm:.4f} ± {cs:.4f}")
    print()

    # --- figura: subplot por tipo, media±std por arquitectura ---
    ncols = 4
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, tipo in enumerate(types):
        ax = axes[i // ncols][i % ncols]
        for p in arch_params:
            means = [agg(p, tipo, n)[0] for n in LEVELS]
            stds = [agg(p, tipo, n)[1] for n in LEVELS]
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            line, = ax.plot(LEVELS, means, marker="o", label=f"{p:,}")
            ax.fill_between(LEVELS, lo, hi, alpha=0.2, color=line.get_color())
        ax.set_title(tipo, fontsize=10)
        ax.set_xticks(LEVELS)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, title="params")
    for j in range(len(types), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Robustez al ruido: media ± std entre semillas, por arquitectura", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115)
    plt.close(fig)

    # --- tabla de deltas de medias (grande - pequena) con veredicto vs ruido de semilla ---
    if len(arch_params) >= 2:
        small, big = arch_params[0], arch_params[-1]
        print(f"delta de MEDIAS ({big:,} - {small:,}); [*] = |delta| supera la suma de stds (robusto a semilla)")
        header = "tipo".ljust(22) + "".join(f"n{n}".rjust(11) for n in LEVELS)
        print(header + "\n" + "-" * len(header))
        for tipo in types:
            cells = tipo.ljust(22)
            for n in LEVELS:
                ms, ss = agg(small, tipo, n)
                mb, sb = agg(big, tipo, n)
                d = mb - ms
                sig = "*" if abs(d) > (ss + sb) else " "
                cells += f"{d:+8.3f}{sig}  "
            print(cells)
    print(f"\nOK: figura en {args.out}")


if __name__ == "__main__":
    main()
