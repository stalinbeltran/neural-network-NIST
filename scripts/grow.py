"""Lanza una ESCALERA de crecimiento gradual de una CNN. CLI delgada -> experiments.growth.

Empieza con una red pequeña y la va agrandando entre etapas (Net2Net), reutilizando lo aprendido.

Uso:  python scripts/grow.py --config configs/growth/cnn_ladder.yaml
"""
from __future__ import annotations

import argparse

from nnist.experiments import run_ladder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    out = run_ladder(args.config)
    print(f"Escalera completada. Comparativa en: {out}")


if __name__ == "__main__":
    main()
