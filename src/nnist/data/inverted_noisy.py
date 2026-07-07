"""Dígitos MNIST INVERTIDOS **y** RUIDOSOS (p. ej. invertido + gaussiano nivel_3).

Combina las dos transformaciones del proyecto en un solo subset: primero el negativo
fotográfico (`invertido`, pixel p -> 1-p; fondo blanco, trazo negro) y **sobre él** un
tipo/nivel de ruido definido en configs/noise/levels.yaml. Es el subset "foto invertida
llena de ruido": mismos dígitos, polaridad opuesta a MNIST y además degradados.

Se materializa igual que los subsets ruidosos (uint8 en disco) y reutiliza el split
train/val CONGELADO vía `load_mnist`, así que TRAIN/VAL/TEST son los mismos índices que en
el resto del proyecto. La semilla del ruido se deriva de (tipo, nivel) pero con un offset
propio para que las realizaciones NO coincidan bit a bit con el subset ruidoso "normal".

  data/processed/inverted_noisy/<tipo>/<nivel>/<split>.pt
  dict: {images uint8 (N,28,28), labels, tipo, param, value, nivel, split, seed, invertido:True}
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from ..utils import set_seed
from .datasets import load_mnist
from .noisy import LEVELS_CFG, _resolve, _subset_seed, load_levels
from .registry import build_transform

INVERTED_NOISY_ROOT = Path("data/processed/inverted_noisy")
_INVERTED_SEED_OFFSET = 50_000   # descorrelaciona el ruido del subset invertido vs. el normal


def generate_inverted_noisy(tipo: str, nivel: str, split: str, *, config_path: str | Path = LEVELS_CFG,
                            root: str | Path = INVERTED_NOISY_ROOT, limit: int | None = None,
                            save: bool = True) -> dict:
    """Genera (y opcionalmente cachea) UN subset invertido+ruidoso. Determinista por (tipo, nivel)."""
    cfg = load_levels(config_path)
    type_idx, param, value, level_num = _resolve(cfg, tipo, nivel)
    seed = _subset_seed(int(cfg.get("seed", 0)), type_idx, level_num) + _INVERTED_SEED_OFFSET

    invert = build_transform("invertido")
    noise = build_transform(tipo, **{param: value})
    transform = lambda img: noise(invert(img))   # invertir primero, ruido encima
    bundle = load_mnist(transform=transform)      # invertido+ruido aplicado + split congelado
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
        "tipo": tipo, "param": param, "value": value, "nivel": nivel, "split": split,
        "seed": seed, "invertido": True,
    }
    if save:
        out_dir = Path(root) / tipo / nivel
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out_dir / f"{split}.pt")
    return blob


def load_inverted_noisy_blob(tipo: str, nivel: str, split: str, *, config_path: str | Path = LEVELS_CFG,
                             root: str | Path = INVERTED_NOISY_ROOT, regenerate: bool = False,
                             limit: int | None = None) -> dict:
    """On-demand: reutiliza el `.pt` si existe; si no, genera el subset y lo cachea."""
    path = Path(root) / tipo / nivel / f"{split}.pt"
    if path.exists() and not regenerate:
        return torch.load(path)
    return generate_inverted_noisy(tipo, nivel, split, config_path=config_path, root=root, limit=limit)


def inverted_noisy_dataset(tipo: str, nivel: str, split: str, **kwargs) -> TensorDataset:
    """Dataset listo para DataLoader: imágenes invertidas+ruidosas en [0,1] (N,1,28,28) + labels."""
    blob = load_inverted_noisy_blob(tipo, nivel, split, **kwargs)
    x = blob["images"].float().div(255.0).unsqueeze(1)
    return TensorDataset(x, blob["labels"])
