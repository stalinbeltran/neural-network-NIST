"""Descarga datasets a data/raw/. CLI delgada.

Uso:  python scripts/download_data.py --dataset mnist
      python scripts/download_data.py --dataset emnist --split digits
"""
from __future__ import annotations

import argparse

from nnist.data import load_emnist, load_mnist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "emnist"])
    ap.add_argument("--split", default="digits", help="(solo emnist) digits|letters|byclass|...")
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()

    if args.dataset == "mnist":
        bundle = load_mnist(root=args.root)
    else:
        bundle = load_emnist(root=args.root, split=args.split)
    print(f"OK: {args.dataset} descargado en {args.root} "
          f"(clases={bundle.num_classes}, input_shape={bundle.input_shape})")


if __name__ == "__main__":
    main()
