"""Pre-calienta (opcional) los SUBSETS RUIDOSOS a disco (CLAUDE.md §1, eje de robustez).

NO es necesario ejecutarlo: los subsets se generan **on-demand** la primera vez que se piden
(ver `nnist.data.noisy_dataset` / `load_noisy_blob`) y se cachean en:

    data/processed/noisy/<tipo>/<nivel>/<split>.pt

Este script solo sirve para MATERIALIZAR por lotes por adelantado (p. ej. antes de un sweep
grande, para no pagar la generación dentro del bucle). Reutiliza la misma lógica que el loader
perezoso, así que produce exactamente los mismos archivos.

Uso:
  python scripts/generate_noisy.py                          # todos los tipos/niveles, split TEST
  python scripts/generate_noisy.py --splits test train val  # también TRAIN y VAL
  python scripts/generate_noisy.py --types gaussiano oclusion --levels 3 5
  python scripts/generate_noisy.py --limit 64               # smoke: 64 imágenes por split
"""
from __future__ import annotations

import argparse

from nnist.data import generate_subset, load_levels
from nnist.data.noisy import LEVELS_CFG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(LEVELS_CFG))
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    ap.add_argument("--types", nargs="+", default=None, help="Filtrar tipos (por defecto: todos).")
    ap.add_argument("--levels", nargs="+", type=int, default=None, help="Filtrar niveles 1..5.")
    ap.add_argument("--limit", type=int, default=None, help="Máx. imágenes por split (smoke test).")
    args = ap.parse_args()

    cfg = load_levels(args.config)
    n_done = 0
    for tname, tdef in cfg["types"].items():
        if args.types and tname not in args.types:
            continue
        for level_name in tdef["levels"]:
            level_num = int(level_name.split("_")[1])
            if args.levels and level_num not in args.levels:
                continue
            for split in args.splits:
                blob = generate_subset(tname, level_name, split,
                                       config_path=args.config, limit=args.limit)
                print(f"  {tname}/{level_name} [{split}] -> {tuple(blob['images'].shape)} "
                      f"({blob['param']}={blob['value']})")
            n_done += 1

    print(f"OK: {n_done} subsets (tipo×nivel) materializados para splits={args.splits}.")


if __name__ == "__main__":
    main()
