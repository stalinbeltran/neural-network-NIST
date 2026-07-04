"""Entrena UNA corrida desde una config. CLI delgada -> delega en experiments.runner.

Uso:  python scripts/train.py --config configs/models/mlp_full.yaml
"""
from __future__ import annotations

import argparse

from nnist.experiments import load_config, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = run(cfg)
    print(result.as_dict())


if __name__ == "__main__":
    main()
