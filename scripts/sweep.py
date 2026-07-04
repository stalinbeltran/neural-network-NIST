"""Lanza un barrido de hiperparámetros. CLI delgada -> delega en experiments.sweep.

Uso:  python scripts/sweep.py --config configs/sweeps/mlp_full_grid.yaml
"""
from __future__ import annotations

import argparse

from nnist.experiments import run_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_sweep(args.config)


if __name__ == "__main__":
    main()
