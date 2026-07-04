"""Expande y ejecuta un BARRIDO de configs para optimizar rendimiento.

Un sweep declara ejes de variación (hiperparámetros, tamaños de subset, arquitecturas) y
produce el producto cartesiano de corridas. Cada corrida usa `runner.run`. Al final agrega
los RunResult para comparar según el criterio deseado (accuracy, accuracy/parámetro, ...).

Formato del YAML de sweep:
    name: <nombre>
    base: configs/models/mlp_full.yaml     # config de partida
    grid:                                   # ejes con claves punteadas
      train.lr: [0.01, 0.001]
      model.hidden: [[64], [128]]
"""
from __future__ import annotations

import copy
import csv
import itertools
from datetime import datetime
from pathlib import Path

import yaml

from .config import from_dict, load_raw, set_by_path
from .runner import run


def expand(sweep_config: dict) -> list:
    """Producto cartesiano de la rejilla -> lista de ExperimentConfig."""
    base = load_raw(sweep_config["base"])
    grid = sweep_config.get("grid", {})
    if not grid:
        return [from_dict(base)]
    keys = list(grid)
    configs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        data = copy.deepcopy(base)
        suffix_parts = []
        for k, v in zip(keys, combo):
            set_by_path(data, k, v)
            suffix_parts.append(f"{k.split('.')[-1]}={v}")
        data["name"] = f"{sweep_config.get('name', base.get('name', 'sweep'))}__" + "_".join(suffix_parts)
        configs.append(from_dict(data))
    return configs


def run_sweep(sweep_config_path: str, output_root: str = "experiments") -> Path:
    """Ejecuta todas las corridas del barrido y agrega resultados en un CSV comparable."""
    with open(sweep_config_path, "r", encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    configs = expand(sweep_config)
    rows = []
    for cfg in configs:
        result = run(cfg, output_root=output_root)
        rows.append({
            "run_id": result.run_id,
            "name": cfg.name,
            "model": result.model_name,
            "strategy": result.strategy,
            "input_shape": "x".join(map(str, result.input_shape)),
            "params_total": result.params_total,
            "accuracy": round(result.accuracy, 4),
            "acc_per_kparam": round(result.accuracy / (result.params_total / 1000), 6)
                              if result.params_total else 0.0,
            "train_seconds": round(result.train_seconds, 2),
        })

    sweep_name = sweep_config.get("name", "sweep")
    out = Path(output_root) / f"_sweep_{sweep_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
