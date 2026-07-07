"""Loaders de datos detrás de una interfaz común.

Se arranca con MNIST/EMNIST (descarga vía torchvision). El diseño deja hueco para NIST SD19
crudo detrás de la MISMA interfaz (CLAUDE.md §3). El nº de clases NO se hardcodea.

`transform` es un callable opcional (p. ej. la ventana de subset de data/transforms.py) que se
aplica sobre el tensor (C, H, W). La `input_shape` reportada es la que ve el modelo, ya recortada.

Protocolo de conjuntos (CLAUDE.md §Datos):
  - TEST: el split de test de fábrica (10.000 en MNIST). NO se toca hasta la evaluación final.
  - VAL:  se recorta del split de entrenamiento (por defecto 20%), estratificado por clase.
  - TRAIN: el resto del split de entrenamiento. Único con el que se ajustan los pesos.

El split train/val se **congela a disco** (data/splits/*.json) la primera vez y se reutiliza tal cual
en todas las corridas y baterías -> la partición es FIJA e idéntica siempre, sin bias aleatorios.
La `split_seed` que lo genera es INDEPENDIENTE del `seed` de entrenamiento (init de pesos, shuffle).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms as T


@dataclass
class DatasetBundle:
    """Lo que un loader devuelve, agnóstico a la fuente."""
    train: object          # torch.utils.data.Dataset (pesos)
    val: object            # torch.utils.data.Dataset (selección/monitorización)
    test: object           # torch.utils.data.Dataset (evaluación final, intocable)
    num_classes: int
    input_shape: tuple[int, ...]   # forma que recibe el modelo (ya con el subset aplicado)


def _compose(transform: Callable | None):
    steps = [T.ToTensor()]                 # PIL -> tensor (1, 28, 28) en [0, 1]
    if transform is not None:
        steps.append(T.Lambda(transform))  # recorte/ventana de subset sobre el tensor
    return T.Compose(steps)


def _infer_shape(dataset) -> tuple[int, ...]:
    x, _ = dataset[0]
    return tuple(x.shape)


def _targets(dataset) -> np.ndarray:
    """Etiquetas del dataset (para estratificar). torchvision expone `.targets`."""
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    return np.array([y for _, y in dataset])   # fallback genérico (lento)


def _fingerprint(targets: np.ndarray) -> str:
    """Huella del dataset: si cambia (tamaño/orden/etiquetas), el split guardado deja de ser válido."""
    return hashlib.sha1(np.ascontiguousarray(targets).tobytes()).hexdigest()[:16]


def _split_indices(targets: np.ndarray, val_fraction: float, split_seed: int):
    """Índices (train, val) estratificados por clase. Determinista para (targets, fraction, seed)."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=split_seed, stratify=targets
    )
    return train_idx.tolist(), val_idx.tolist()


def stratified_split(dataset, val_fraction: float, split_seed: int) -> tuple[Subset, Subset]:
    """Split estratificado EN MEMORIA (sin persistir). Útil para tests / usos puntuales."""
    train_idx, val_idx = _split_indices(_targets(dataset), val_fraction, split_seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def frozen_stratified_split(dataset, val_fraction: float, split_seed: int,
                            cache_path: str | Path) -> tuple[Subset, Subset]:
    """Split estratificado CONGELADO a disco: se calcula una vez y se reutiliza idéntico siempre.

    - Si `cache_path` existe: carga los índices y verifica la huella del dataset. Si no coincide
      (dataset distinto o `val_fraction` distinta), lanza error en vez de usar un split incorrecto.
    - Si no existe: lo calcula, lo guarda y lo devuelve. Commitea ese archivo para compartir el
      MISMO split entre máquinas y baterías.
    """
    targets = _targets(dataset)
    fp = _fingerprint(targets)
    cache_path = Path(cache_path)

    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fp or data.get("val_fraction") != val_fraction:
            raise RuntimeError(
                f"El split congelado en {cache_path} no coincide con el dataset/parámetros actuales "
                f"(fingerprint o val_fraction distintos). Bórralo a conciencia para regenerarlo."
            )
        train_idx, val_idx = data["train_idx"], data["val_idx"]
    else:
        train_idx, val_idx = _split_indices(targets, val_fraction, split_seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "n": len(targets),
            "fingerprint": fp,
            "val_fraction": val_fraction,
            "split_seed": split_seed,
            "train_idx": train_idx,
            "val_idx": val_idx,
        }), encoding="utf-8")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _split_cache(splits_dir: str, key: str, val_fraction: float, split_seed: int) -> Path:
    return Path(splits_dir) / f"{key}_val{val_fraction}_seed{split_seed}.json"


def load_mnist(root: str = "data/raw", transform: Callable | None = None,
               val_fraction: float = 0.2, split_seed: int = 0,
               splits_dir: str = "data/splits") -> DatasetBundle:
    tf = _compose(transform)
    full_train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    cache = _split_cache(splits_dir, "mnist", val_fraction, split_seed)
    train, val = frozen_stratified_split(full_train, val_fraction, split_seed, cache)
    return DatasetBundle(train, val, test, num_classes=10, input_shape=_infer_shape(train))


# EMNIST tiene varios splits; cada uno fija el nº de clases (digits=10, letters=26, byclass=62...).
_EMNIST_CLASSES = {"digits": 10, "mnist": 10, "letters": 26, "balanced": 47, "byclass": 62, "bymerge": 47}


def load_emnist(root: str = "data/raw", split: str = "digits", transform: Callable | None = None,
                val_fraction: float = 0.2, split_seed: int = 0,
                splits_dir: str = "data/splits") -> DatasetBundle:
    tf = _compose(transform)
    full_train = datasets.EMNIST(root, split=split, train=True, download=True, transform=tf)
    test = datasets.EMNIST(root, split=split, train=False, download=True, transform=tf)
    num_classes = _EMNIST_CLASSES.get(split)
    if num_classes is None:
        raise ValueError(f"split EMNIST desconocido: {split!r}")
    cache = _split_cache(splits_dir, f"emnist-{split}", val_fraction, split_seed)
    train, val = frozen_stratified_split(full_train, val_fraction, split_seed, cache)
    return DatasetBundle(train, val, test, num_classes=num_classes, input_shape=_infer_shape(train))


def load_nist_sd19(root: str = "data/raw/sd19", transform: Callable | None = None,
                   val_fraction: float = 0.2, split_seed: int = 0,
                   splits_dir: str = "data/splits") -> DatasetBundle:
    """TODO: NIST Special Database 19 crudo (PNG por clase) detrás de la misma interfaz."""
    raise NotImplementedError("load_nist_sd19 pendiente")


def _load_lines_curves(**kwargs) -> DatasetBundle:
    from .shapes import load_lines_curves   # import perezoso (evita ciclo: shapes importa de aquí)
    return load_lines_curves(**kwargs)


def _load_lines_hv(**kwargs) -> DatasetBundle:
    from .shapes import load_lines_hv
    return load_lines_hv(**kwargs)


_LOADERS = {"mnist": load_mnist, "emnist": load_emnist, "nist_sd19": load_nist_sd19,
            "lines_curves": _load_lines_curves, "lines_hv": _load_lines_hv}


def load_dataset(name: str, transform: Callable | None = None, **kwargs) -> DatasetBundle:
    """Dispatch por nombre desde la config. `kwargs` extra (val_fraction, split_seed, split...) pasan al loader."""
    if name not in _LOADERS:
        raise KeyError(f"Dataset desconocido: {name!r}. Disponibles: {sorted(_LOADERS)}")
    return _LOADERS[name](transform=transform, **kwargs)
