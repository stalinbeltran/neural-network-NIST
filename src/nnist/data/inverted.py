"""Dataset de dígitos MNIST INVERTIDOS (negativo fotográfico: pixel p -> 1 - p).

Los mismos dígitos originales, con fondo y trazo intercambiados (fondo blanco, trazo negro).
No es ruido ni tiene niveles: es una transformación determinista y reversible. Se materializa
igual que los subsets ruidosos (uint8 en disco), reutilizando el split train/val CONGELADO vía
`load_mnist`, y se genera on-demand con caché.

  data/processed/inverted/<split>.pt   (dict: images uint8 (N,28,28), labels, transform, split)
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from .datasets import load_mnist
from .registry import build_transform

INVERTED_ROOT = Path("data/processed/inverted")


def generate_inverted(split: str, *, root: str | Path = INVERTED_ROOT,
                      limit: int | None = None, save: bool = True) -> dict:
    """Genera (y cachea) el split invertido. Determinista: inversión = 1 - pixel."""
    tf = build_transform("invertido")
    bundle = load_mnist(transform=tf)          # inversión aplicada + split congelado
    ds = getattr(bundle, split)
    imgs, labels = [], []
    for i, (x, y) in enumerate(ds):
        if limit is not None and i >= limit:
            break
        imgs.append((x[0] * 255).round().clamp(0, 255).to(torch.uint8))
        labels.append(int(y))
    blob = {"images": torch.stack(imgs), "labels": torch.tensor(labels, dtype=torch.long),
            "transform": "invertido", "split": split}
    if save:
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out / f"{split}.pt")
    return blob


def load_inverted(split: str, *, root: str | Path = INVERTED_ROOT,
                  regenerate: bool = False, limit: int | None = None) -> dict:
    """On-demand: reutiliza el `.pt` si existe; si no, genera el split invertido y lo cachea."""
    path = Path(root) / f"{split}.pt"
    if path.exists() and not regenerate:
        return torch.load(path)
    return generate_inverted(split, root=root, limit=limit)


def inverted_dataset(split: str, **kwargs) -> TensorDataset:
    """Dataset listo para DataLoader: imágenes invertidas en [0,1] con forma (N,1,28,28) + labels."""
    blob = load_inverted(split, **kwargs)
    x = blob["images"].float().div(255.0).unsqueeze(1)
    return TensorDataset(x, blob["labels"])
