"""Entrena UNA corrida desde una config. CLI delgada -> delega en experiments.runner.

Uso:
  python scripts/train.py --config configs/models/mlp_full.yaml
  python scripts/train.py --config configs/models/mlp_full.yaml --resume experiments/<run_dir>

Para reanudar: la corrida original debe haberse lanzado con `checkpoint_every` en la config, y
`--resume` apunta a su carpeta (con checkpoint.pt). Sube `train.epochs` en la config para entrenar más.
"""
from __future__ import annotations

import argparse

from nnist.experiments import load_config, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="carpeta de una corrida previa con checkpoint.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = run(cfg, resume_from=args.resume)
    print(result.as_dict())


if __name__ == "__main__":
    main()
