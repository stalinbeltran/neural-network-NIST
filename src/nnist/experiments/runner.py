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
from ..training import EarlyStopping, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger, find_lr
from ..utils import get_logger, log_training, set_seed

logger = get_logger("nnist.runner")


def _build_transform(cfg_transform: dict | None):
    """Devuelve (callable|None, strategy_str). La estrategia de subset es un transform."""
    if not cfg_transform:
        return None, "full"
    params = dict(cfg_transform)
    name = params.pop("name")
    return build_transform(name, **params), f"subset:{name}"


def run(config, output_root: str = "experiments", resume_from: str | None = None,
        prebuilt_model=None, return_model: bool = False):
    """Ejecuta (o REANUDA) una corrida. Devuelve un `RunResult` (o `(RunResult, model)` si
    `return_model=True`).

    `resume_from`: carpeta de una corrida previa con `checkpoint.pt`. Se continúa el entrenamiento
    en esa misma carpeta hasta `config.train.epochs` (que debe ser >= la época del checkpoint).
    El checkpoint se guarda cada `config.train.checkpoint_every` épocas (opt-in; por defecto 5 al reanudar).

    `prebuilt_model`: si se pasa, se usa ESE modelo en vez de construirlo desde `config.model` (lo
    usa el crecimiento gradual, que trasplanta pesos con `grow_cnn` antes de entrenar). Debe tener la
    `input_shape`/`num_classes` que derivan de los datos.
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
    if prebuilt_model is not None:
        if tuple(prebuilt_model.input_shape) != tuple(bundle.input_shape):
            raise ValueError(f"prebuilt_model.input_shape {tuple(prebuilt_model.input_shape)} != "
                             f"datos {tuple(bundle.input_shape)}")
        if prebuilt_model.num_classes != bundle.num_classes:
            raise ValueError(f"prebuilt_model.num_classes {prebuilt_model.num_classes} != "
                             f"datos {bundle.num_classes}")
        model = prebuilt_model
    else:
        model = build_model(config.model["name"], input_shape=bundle.input_shape,
                            num_classes=bundle.num_classes, **model_kwargs)
    params = model.count_params()
    logger.info("Modelo %s | params=%s", config.model["name"], params)

    # 3. entrenamiento (pesos con TRAIN, monitorización/selección con VAL)
    train_dict = dict(config.train)
    checkpoint_every = train_dict.pop("checkpoint_every", None)   # opt-in; no es campo de TrainConfig
    early_stop = train_dict.pop("early_stopping", None)           # opt-in; int (patience) o dict

    # lr: auto -> el LR finder elige el learning rate inicial automáticamente (sondea unos batches
    # subiendo el lr y coge el del descenso más pronunciado; restaura los pesos, entrena desde cero).
    auto_lr = None
    if train_dict.get("lr") == "auto":
        probe = DataLoader(bundle.train, batch_size=train_dict.get("batch_size", 128), shuffle=True)
        auto_lr = find_lr(model, probe, optimizer=train_dict.get("optimizer", "adam"),
                          weight_decay=train_dict.get("weight_decay", 0.0),
                          device=train_dict.get("device", "cpu"))["suggested_lr"]
        train_dict["lr"] = auto_lr
        logger.info("LR automático (LR finder): %.4g", auto_lr)

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

    # bitácora única de entrenamientos (trainings/TRAININGS.md): estado en_curso -> hecho
    modelo_txt = f"{config.model['name']} {model_kwargs}" if model_kwargs else config.model["name"]
    ckpt_txt = str(out_dir / "checkpoint.pt") if checkpoint_every else str(out_dir / "model.pt")
    callbacks = [TrainingLogger(run_id, train_cfg.epochs, modelo=modelo_txt,
                                datos=f"{config.dataset['name']} ({strategy})", checkpoint=ckpt_txt)]
    if checkpoint_every:
        callbacks.append(ModelCheckpoint(out_dir / "checkpoint.pt", every=checkpoint_every))
    if early_stop is not None:
        es_kwargs = {"patience": early_stop} if isinstance(early_stop, int) else dict(early_stop)
        callbacks.append(EarlyStopping(**es_kwargs))
    trainer = Trainer(model, train_cfg, callbacks=callbacks)

    if resume_from:
        ckpt = trainer.resume_from(out_dir / "checkpoint.pt")
        logger.info("Reanudando %s desde época %d hasta %d", run_id, ckpt["epochs_done"], train_cfg.epochs)

    log_training(id=run_id, estado="en_curso", modelo=modelo_txt,
                 datos=f"{config.dataset['name']} ({strategy})",
                 épocas=f"{trainer.start_epoch}/{train_cfg.epochs}", checkpoint=ckpt_txt)
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
        extra={"classification_report": report,
               **({"auto_lr": auto_lr} if auto_lr is not None else {})},
    )

    # 5. persistir corrida reproducible (out_dir ya creada antes del entrenamiento)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)   # curva de aprendizaje por época (para graficar)
    torch.save(model.state_dict(), out_dir / "model.pt")
    log_training(id=run_id, estado="hecho", épocas=f"{trainer._epochs_done}/{train_cfg.epochs}",
                 val=val_acc, test=test_acc)
    logger.info("Corrida guardada en %s | val=%.4f | test=%.4f", out_dir, val_acc, test_acc)
    return (result, model) if return_model else result
