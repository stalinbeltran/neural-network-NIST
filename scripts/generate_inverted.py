"""Materializa el dataset de dígitos MNIST INVERTIDOS (negativo: pixel -> 1 - pixel).

Los mismos dígitos originales con fondo/trazo intercambiados. Reutiliza el split congelado.
Se genera también on-demand vía `nnist.data.inverted_dataset(split)`; este script lo pre-calienta.

  data/processed/inverted/<split>.pt

Uso:
  python scripts/generate_inverted.py                     # train, val y test
  python scripts/generate_inverted.py --splits test
"""
from __future__ import annotations

import argparse

from nnist.data import generate_inverted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for split in args.splits:
        blob = generate_inverted(split, limit=args.limit)
        print(f"  invertido [{split}] -> {tuple(blob['images'].shape)}")
    print(f"OK: dataset invertido en data/processed/inverted/ (splits={args.splits}).")


if __name__ == "__main__":
    main()
