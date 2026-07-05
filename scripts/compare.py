"""Compara dos baterías OFAT (p. ej. MLP@10 vs CNN@10) a partir de sus CSV de resumen.

Uso:
  python scripts/compare.py --a ofat_mlp_full_10ep --b ofat_cnn_full --metric test_mean
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nnist.viz import plot_axis_effects, plot_comparison


def latest_summary(name: str, root: str = "experiments") -> Path | None:
    files = sorted(Path(root).glob(f"_ofat_{name}_*_summary.csv"))
    return files[-1] if files else None


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best(rows: list[dict], metric: str) -> dict:
    return max(rows, key=lambda r: float(r[metric]))


def _baseline(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r["group"] == "baseline"), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="nombre del sweep A (p. ej. ofat_mlp_full_10ep)")
    ap.add_argument("--b", required=True, help="nombre del sweep B (p. ej. ofat_cnn_full)")
    ap.add_argument("--metric", default="test_mean", choices=["test_mean", "val_mean"])
    ap.add_argument("--outdir", default="experiments/_plots")
    args = ap.parse_args()

    a_path, b_path = latest_summary(args.a), latest_summary(args.b)
    if not a_path or not b_path:
        print(f"Falta el summary de {args.a if not a_path else args.b}")
        return
    a, b = load(a_path), load(b_path)
    std_key = args.metric.replace("_mean", "_std")

    out = Path(args.outdir) / f"compare_{args.a}_vs_{args.b}.png"
    plot_comparison({args.a: a, args.b: b}, metric=args.metric, std_key=std_key, out=str(out))
    print(f"Guardado: {out}")

    out_axes = Path(args.outdir) / f"axes_{args.a}_vs_{args.b}.png"
    plot_axis_effects({args.a: a, args.b: b}, metric=args.metric, out=str(out_axes))
    print(f"Guardado: {out_axes}\n")

    for name, rows in ((args.a, a), (args.b, b)):
        base, best = _baseline(rows), _best(rows, args.metric)
        bl = f"{base['group']}={base[args.metric]} ({base['params_total']}p)" if base else "n/a"
        print(f"[{name}]  baseline: {bl}")
        print(f"          mejor:    {best['group']}={best[args.metric]} ({best['params_total']}p)")


if __name__ == "__main__":
    main()
