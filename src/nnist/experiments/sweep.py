"""Barridos de experimentos. Soporta dos modos:

1) GRID (producto cartesiano de ejes):
    name: <nombre>
    base: configs/models/mlp_full.yaml
    grid:
      train.lr: [0.01, 0.001]
      model.hidden: [[64], [128]]

2) OFAT (un factor a la vez): parte de un `baseline` y varía UN eje cada vez, dejando el resto
   fijo. Aísla el efecto de cada hiperparámetro. Cada configuración se repite con varias `seeds`
   (init de pesos; el split de datos sigue congelado e idéntico) y se agrega mean±std.
    name: <nombre>
    base: configs/models/mlp_full.yaml
    mode: ofat
    baseline:
      train.lr: 0.003
      model.hidden: [128]
    axes:
      train.lr: [0.01, 0.001]
      model.hidden: [[64], [256, 128]]
    seeds: [0, 1, 2, 3, 4]
"""
from __future__ import annotations

import copy
import csv
import itertools
import statistics
from datetime import datetime
from pathlib import Path

import yaml

from .config import from_dict, load_raw, set_by_path
from .runner import run


# --------------------------------------------------------------------------- helpers

def _apply(base: dict, overrides: dict) -> dict:
    data = copy.deepcopy(base)
    for k, v in overrides.items():
        set_by_path(data, k, v)
    return data


def _fmt(v) -> str:
    """Etiqueta legible y segura para nombres de archivo a partir de un valor de config."""
    return str(v).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")


def _row(group: str, seed, cfg, result) -> dict:
    return {
        "group": group,
        "seed": seed,
        "run_id": result.run_id,
        "model": result.model_name,
        "strategy": result.strategy,
        "input_shape": "x".join(map(str, result.input_shape)),
        "params_total": result.params_total,
        "val_accuracy": round(result.val_accuracy, 4),
        "test_accuracy": round(result.accuracy, 4),
        "train_seconds": round(result.train_seconds, 2),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- GRID

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
        parts = []
        for k, v in zip(keys, combo):
            set_by_path(data, k, v)
            parts.append(f"{k.split('.')[-1]}={_fmt(v)}")
        data["name"] = f"{sweep_config.get('name', base.get('name', 'sweep'))}__" + "_".join(parts)
        configs.append(from_dict(data))
    return configs


def run_grid(sweep_config: dict, output_root: str) -> Path:
    rows = []
    for cfg in expand(sweep_config):
        result = run(cfg, output_root=output_root)
        rows.append(_row("grid", cfg.seed, cfg, result))
    rows.sort(key=lambda r: r["val_accuracy"], reverse=True)
    out = Path(output_root) / f"_grid_{sweep_config.get('name', 'sweep')}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    _write_csv(out, rows)
    return out


# --------------------------------------------------------------------------- OFAT

def expand_ofat(spec: dict) -> list[tuple[str, object]]:
    """Devuelve [(group, ExperimentConfig)]: baseline + una variación por (eje, valor), cada
    una repetida por cada seed. `group` identifica la configuración (compartido entre semillas)."""
    base = load_raw(spec["base"])
    baseline = spec.get("baseline", {})
    axes = spec.get("axes", {})
    seeds = spec.get("seeds", [0])
    name = spec.get("name", "ofat")

    base_cfg = _apply(base, baseline)   # baseline aplicado sobre la base

    cells: list[tuple[str, dict]] = [("baseline", {})]
    for key, values in axes.items():
        short = key.split(".")[-1]
        for v in values:
            if key in baseline and baseline[key] == v:
                continue   # no repetir el baseline
            cells.append((f"{short}={_fmt(v)}", {key: v}))

    runs = []
    for group, override in cells:
        cell = _apply(base_cfg, override)
        for seed in seeds:
            data = copy.deepcopy(cell)
            data["seed"] = seed
            data["name"] = f"{name}__{group}__seed{seed}"
            runs.append((group, from_dict(data)))
    return runs


def _aggregate(rows: list[dict]) -> list[dict]:
    """Agrupa por `group` (config) y calcula media y desviación de val/test sobre las semillas."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    summary = []
    for group, rs in groups.items():
        val = [r["val_accuracy"] for r in rs]
        test = [r["test_accuracy"] for r in rs]
        params = rs[0]["params_total"]
        summary.append({
            "group": group,
            "params_total": params,
            "n_seeds": len(rs),
            "val_mean": round(statistics.mean(val), 4),
            "val_std": round(statistics.pstdev(val), 4) if len(val) > 1 else 0.0,
            "test_mean": round(statistics.mean(test), 4),
            "test_std": round(statistics.pstdev(test), 4) if len(test) > 1 else 0.0,
            "val_mean_per_kparam": round(statistics.mean(val) / (params / 1000), 6) if params else 0.0,
        })
    summary.sort(key=lambda r: r["val_mean"], reverse=True)
    return summary


def run_ofat(spec: dict, output_root: str) -> tuple[Path, Path]:
    """Ejecuta la batería OFAT. Devuelve (csv_por_corrida, csv_resumen_agregado)."""
    runs = expand_ofat(spec)
    rows = []
    for group, cfg in runs:
        result = run(cfg, output_root=output_root)
        rows.append(_row(group, cfg.seed, cfg, result))

    name = spec.get("name", "ofat")
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    runs_csv = Path(output_root) / f"_ofat_{name}_{stamp}_runs.csv"
    summary_csv = Path(output_root) / f"_ofat_{name}_{stamp}_summary.csv"
    _write_csv(runs_csv, rows)
    _write_csv(summary_csv, _aggregate(rows))
    return runs_csv, summary_csv


# --------------------------------------------------------------------------- dispatch

def run_sweep(sweep_config_path: str, output_root: str = "experiments"):
    """Detecta el modo (grid u ofat) y ejecuta. Devuelve el/los CSV de resultados."""
    with open(sweep_config_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if spec.get("mode") == "ofat" or "axes" in spec:
        return run_ofat(spec, output_root)
    return run_grid(spec, output_root)
