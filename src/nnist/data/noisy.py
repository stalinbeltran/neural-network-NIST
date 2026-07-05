"""Subsets ruidosos derivados de MNIST, con GENERACIÓN PEREZOSA (on-demand).

No hace falta materializar los 55 subsets por adelantado: el ruido es un transform
determinista (misma semilla -> mismas imágenes), así que un subset se genera la PRIMERA
vez que se pide y se cachea en disco; las siguientes veces se reutiliza. Solo existe en
disco lo que realmente se use.

Layout de la caché (gitignored):  data/processed/noisy/<tipo>/<nivel>/<split>.pt
Cada `.pt` = dict {images: uint8 (N,28,28), labels: int64 (N,), tipo, param, value, nivel, split, seed}.

La definición de tipos/niveles vive en configs/noise/levels.yaml (única fuente de verdad).
Reutiliza el split train/val CONGELADO vía `load_mnist`, de modo que TRAIN/VAL/TEST son los
mismos índices que en el resto del proyecto. La semilla del ruido se deriva de (tipo, nivel).
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import TensorDataset

from ..utils import set_seed
from .datasets import load_mnist
from .registry import build_transform

NOISY_ROOT = Path("data/processed/noisy")
LEVELS_CFG = Path("configs/noise/levels.yaml")


def load_levels(config_path: str | Path = LEVELS_CFG) -> dict:
    """Lee configs/noise/levels.yaml (definición de tipos × niveles)."""
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _subset_seed(base: int, type_idx: int, level: int) -> int:
    """Semilla estable y distinta por (tipo, nivel): ruido reproducible y descorrelacionado."""
    return base * 100_000 + type_idx * 100 + level


def _resolve(cfg: dict, tipo: str, nivel: str):
    """Devuelve (type_idx, param, value, level_num) para (tipo, nivel) del config."""
    types = cfg["types"]
    if tipo not in types:
        raise KeyError(f"Tipo de ruido desconocido: {tipo!r}. Disponibles: {sorted(types)}")
    tdef = types[tipo]
    if nivel not in tdef["levels"]:
        raise KeyError(f"Nivel desconocido para {tipo!r}: {nivel!r}. Disponibles: {sorted(tdef['levels'])}")
    return list(types).index(tipo), tdef["param"], tdef["levels"][nivel], int(nivel.split("_")[1])


def generate_subset(tipo: str, nivel: str, split: str, *, config_path: str | Path = LEVELS_CFG,
                    root: str | Path = NOISY_ROOT, limit: int | None = None, save: bool = True) -> dict:
    """Genera (y opcionalmente cachea) UN subset ruidoso. Determinista por (tipo, nivel)."""
    cfg = load_levels(config_path)
    type_idx, param, value, level_num = _resolve(cfg, tipo, nivel)
    seed = _subset_seed(int(cfg.get("seed", 0)), type_idx, level_num)

    transform = build_transform(tipo, **{param: value})
    bundle = load_mnist(transform=transform)     # ruido aplicado + split congelado
    ds = getattr(bundle, split)

    set_seed(seed)
    imgs, labels = [], []
    for i, (x, y) in enumerate(ds):
        if limit is not None and i >= limit:
            break
        imgs.append((x[0] * 255).round().clamp(0, 255).to(torch.uint8))
        labels.append(int(y))

    blob = {
        "images": torch.stack(imgs), "labels": torch.tensor(labels, dtype=torch.long),
        "tipo": tipo, "param": param, "value": value, "nivel": nivel, "split": split, "seed": seed,
    }
    if save:
        out_dir = Path(root) / tipo / nivel
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out_dir / f"{split}.pt")
    return blob


def load_noisy_blob(tipo: str, nivel: str, split: str, *, config_path: str | Path = LEVELS_CFG,
                    root: str | Path = NOISY_ROOT, regenerate: bool = False,
                    limit: int | None = None) -> dict:
    """On-demand: reutiliza el `.pt` si existe; si no, genera el subset y lo cachea.

    `limit` (solo al generar) recorta el nº de imágenes — útil para smoke tests.
    """
    path = Path(root) / tipo / nivel / f"{split}.pt"
    if path.exists() and not regenerate:
        return torch.load(path)
    return generate_subset(tipo, nivel, split, config_path=config_path, root=root, limit=limit)


def noisy_dataset(tipo: str, nivel: str, split: str, **kwargs) -> TensorDataset:
    """Dataset listo para DataLoader: imágenes en [0,1] con forma (N,1,28,28) + labels."""
    blob = load_noisy_blob(tipo, nivel, split, **kwargs)
    x = blob["images"].float().div(255.0).unsqueeze(1)
    return TensorDataset(x, blob["labels"])
