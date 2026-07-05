"""Ejecuta UNA corrida a partir de una ExperimentConfig y guarda un resultado reproducible.

Ensambla: dataset -> transform (subset opcional) -> modelo (input_shape derivada) -> Trainer.
Persiste en experiments/<run_id>/ {config.yaml, metrics.json, model.pt}. Ver CLAUDE.md §5.5.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..data import build_transform, load_dataset
from ..evaluation import RunResult, classification_report
from ..models import build_model
from ..training import ModelCheckpoint, TrainConfig, Trainer
from ..utils import get_logger, set_seed

logger = get_logger("nnist.runner")


def _build_transform(cfg_transform: dict | None):
    """Devuelve (callable|None, strategy_str). La estrategia de subset es un transform."""
    if not cfg_transform:
        return None, "full"
    params = dict(cfg_transform)
    name = params.pop("name")
    return build_transform(name, **params), f"subset:{name}"


def run(config, output_root: str = "experiments", resume_from: str | None = None) -> RunResult:
    """Ejecuta (o REANUDA) una corrida.

    `resume_from`: carpeta de una corrida previa con `checkpoint.pt`. Se continúa el entrenamiento
    en esa misma carpeta hasta `config.train.epochs` (que debe ser >= la época del checkpoint).
    El checkpoint se guarda cada `config.train.checkpoint_every` épocas (opt-in; por defecto 5 al reanudar).
    """
    set_seed(config.seed)

    # 1. datos + estrategia (imagen completa o subset)
    transform, strategy = _build_transform(config.transform)
    # El split train/val NO depende de config.seed (init de pesos): usa su propia split_seed FIJA,
    # congelada a disco, para que la partición sea idéntica en todas las corridas y baterías.
    ds_kwargs = {k: v for k, v in config.dataset.items() if k != "name"}
    bundle = load_dataset(config.dataset["name"], transform=transform, **ds_kwargs)
    logger.info("Dataset %s | clases=%d | input_shape=%s | estrategia=%s",
                config.dataset["name"], bundle.num_classes, bundle.input_shape, strategy)

    # 2. modelo, con input_shape DERIVADA de los datos (no hardcodeada)
    model_kwargs = {k: v for k, v in config.model.items() if k != "name"}
    model = build_model(config.model["name"], input_shape=bundle.input_shape,
                        num_classes=bundle.num_classes, **model_kwargs)
    params = model.count_params()
    logger.info("Modelo %s | params=%s", config.model["name"], params)

    # 3. entrenamiento (pesos con TRAIN, monitorización/selección con VAL)
    train_dict = dict(config.train)
    checkpoint_every = train_dict.pop("checkpoint_every", None)   # opt-in; no es campo de TrainConfig
    train_cfg = TrainConfig(**train_dict)
    train_loader = DataLoader(bundle.train, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(bundle.test, batch_size=train_cfg.batch_size, shuffle=False)

    # carpeta de la corrida: al reanudar reutilizamos la existente; si no, una nueva con timestamp
    if resume_from:
        out_dir = Path(resume_from)
        run_id = out_dir.name
        if checkpoint_every is None:
            checkpoint_every = 5      # al reanudar, seguir guardando por defecto
    else:
        run_id = f"{config.name}_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if checkpoint_every:
        callbacks.append(ModelCheckpoint(out_dir / "checkpoint.pt", every=checkpoint_every))
    trainer = Trainer(model, train_cfg, callbacks=callbacks)

    if resume_from:
        ckpt = trainer.resume_from(out_dir / "checkpoint.pt")
        logger.info("Reanudando %s desde época %d hasta %d", run_id, ckpt["epochs_done"], train_cfg.epochs)

    t0 = time.perf_counter()
    history = trainer.fit(train_loader, val_loader)
    train_seconds = time.perf_counter() - t0
    val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0

    # 4. evaluación FINAL sobre TEST (intocado durante el entrenamiento y la selección)
    test_acc, y_true, y_pred, infer_ms = trainer.evaluate(test_loader)
    try:
        report = classification_report(y_true, y_pred)
    except Exception as e:  # sklearn ausente u otro problema no debe tumbar la corrida
        logger.warning("classification_report no disponible: %s", e)
        report = {}

    result = RunResult(
        run_id=run_id,
        model_name=config.model["name"],
        strategy=strategy,
        input_shape=tuple(bundle.input_shape),
        num_classes=bundle.num_classes,
        params_total=params["params_total"],
        params_trainable=params["params_trainable"],
        accuracy=test_acc,          # número honesto final (TEST)
        val_accuracy=val_acc,       # métrica de selección/comparación del sweep (VAL)
        train_seconds=train_seconds,
        infer_ms_per_sample=infer_ms,
        extra={"classification_report": report},
    )

    # 5. persistir corrida reproducible (out_dir ya creada antes del entrenamiento)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)   # curva de aprendizaje por época (para graficar)
    torch.save(model.state_dict(), out_dir / "model.pt")
    logger.info("Corrida guardada en %s | val=%.4f | test=%.4f", out_dir, val_acc, test_acc)
    return result
