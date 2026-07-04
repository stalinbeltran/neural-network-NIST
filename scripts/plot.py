"""Genera gráficas a partir de las corridas en experiments/. CLI delgada.

Uso:
  python scripts/plot.py                          # todas las corridas
  python scripts/plot.py --pattern mlp_full_grid  # solo las de un sweep
  python scripts/plot.py --kind curves            # solo curvas de aprendizaje
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nnist.viz import load_runs, plot_learning_curves, plot_pareto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="experiments")
    ap.add_argument("--pattern", default=None, help="subcadena del run_id (p. ej. el nombre del sweep)")
    ap.add_argument("--kind", default="all", choices=["all", "curves", "pareto"])
    ap.add_argument("--metric", default="val_accuracy", help="serie de history para las curvas")
    ap.add_argument("--outdir", default="experiments/_plots")
    args = ap.parse_args()

    runs = load_runs(args.experiments, pattern=args.pattern)
    if not runs:
        print("No se encontraron corridas.")
        return
    print(f"{len(runs)} corridas cargadas.")

    tag = args.pattern or "all"
    outdir = Path(args.outdir)
    if args.kind in ("all", "curves"):
        p = plot_learning_curves(runs, metric=args.metric, out=str(outdir / f"curvas_{tag}.png"))
        print(f"Guardado: {p}")
    if args.kind in ("all", "pareto"):
        p = plot_pareto(runs, out=str(outdir / f"pareto_{tag}.png"))
        print(f"Guardado: {p}")


if __name__ == "__main__":
    main()
