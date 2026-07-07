"""Escalera de crecimiento: entrena una CNN por ETAPAS, agrandándola entre etapa y etapa.

Cada etapa entrena una `SimpleCNN`; la siguiente parte de los pesos de la anterior trasplantados
con `grow_cnn` (Net2Net) y añade capacidad (más canales y/o más bloques y/o cabeza densa mayor).
Así se empieza con una red pequeña (rápida de entrenar) y se sube capacidad reutilizando lo aprendido.

Spec (YAML), ver `configs/growth/cnn_ladder.yaml`:

    name: cnn_ladder
    base: configs/models/cnn_full.yaml   # config CNN de partida (dataset, kernel, batchnorm...)
    seed: 0
    train: {epochs: 3, lr: 0.001}        # defaults de entrenamiento por etapa (opcional)
    stages:
      - {channels: [16],    fc_hidden: 64}            # etapa 0: entrena desde cero
      - {channels: [32],    fc_hidden: 64}            # ensancha conv (exacto, sin bache)
      - {channels: [32, 64], fc_hidden: 64}           # profundiza (warm-start, bache pequeño)
      - {channels: [32, 64], fc_hidden: 128, epochs: 5}  # ensancha la densa (exacto)

Cada etapa es una corrida independiente en `experiments/` y una fila en `trainings/TRAININGS.md`;
al final se emite un CSV comparando accuracy vs. nº de parámetros por etapa.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime
from pathlib import Path

import yaml

from ..models import grow_cnn
from .config import from_dict, load_raw, set_by_path
from .runner import run

_GROW_KEYS = {"channels": "model.channels", "fc_hidden": "model.fc_hidden"}
_TRAIN_KEYS = {"epochs": "train.epochs", "lr": "train.lr", "batch_size": "train.batch_size"}


def _stage_config(base: dict, spec: dict, stage: dict, index: int, seed: int):
    """Construye la ExperimentConfig de una etapa: base + defaults de train + overrides de la etapa."""
    data = copy.deepcopy(base)
    for key, value in spec.get("train", {}).items():
        set_by_path(data, _TRAIN_KEYS.get(key, f"train.{key}"), value)
    for key, value in stage.items():
        if key in _GROW_KEYS:
            set_by_path(data, _GROW_KEYS[key], value)
        elif key in _TRAIN_KEYS:
            set_by_path(data, _TRAIN_KEYS[key], value)
        else:
            set_by_path(data, key, value)
    data["seed"] = seed
    data["name"] = f"{spec.get('name', 'ladder')}__s{index}"
    return from_dict(data)


def run_ladder(spec_path: str, output_root: str = "experiments") -> Path:
    """Ejecuta la escalera definida en `spec_path`. Devuelve la ruta del CSV comparativo."""
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    base = load_raw(spec["base"])
    if base.get("model", {}).get("name") != "cnn":
        raise ValueError("La escalera de crecimiento solo soporta model.name == 'cnn'.")
    seed = spec.get("seed", 0)
    stages = spec["stages"]

    rows = []
    prev_model = None
    for i, stage in enumerate(stages):
        cfg = _stage_config(base, spec, stage, i, seed)
        m = cfg.model
        if prev_model is None:                       # etapa 0: entrena desde cero
            result, prev_model = run(cfg, output_root=output_root, return_model=True)
        else:                                        # etapas siguientes: crecer desde la anterior
            grown = grow_cnn(prev_model, channels=m.get("channels"), fc_hidden=m.get("fc_hidden"),
                             seed=seed)
            result, prev_model = run(cfg, output_root=output_root, prebuilt_model=grown,
                                     return_model=True)
        rows.append({
            "stage": i,
            "run_id": result.run_id,
            "channels": "x".join(map(str, m.get("channels", []))),
            "fc_hidden": m.get("fc_hidden"),
            "params_total": result.params_total,
            "val_accuracy": round(result.val_accuracy, 4),
            "test_accuracy": round(result.accuracy, 4),
            "val_per_kparam": round(result.val_accuracy / (result.params_total / 1000), 6)
            if result.params_total else 0.0,
            "train_seconds": round(result.train_seconds, 2),
        })

    out = Path(output_root) / f"_ladder_{spec.get('name', 'ladder')}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
