"""Visualización de resultados de corridas y barridos.

Lee las carpetas experiments/<run_id>/ (metrics.json + history.json) y produce gráficas:
  - curvas de aprendizaje (accuracy/loss por época, una línea por corrida)
  - frontera de Pareto (accuracy final vs nº de parámetros)

Usa el backend "Agg" (sin ventana) para guardar PNGs de forma robusta en cualquier entorno.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Run:
    run_id: str
    metrics: dict           # contenido de metrics.json
    history: dict           # contenido de history.json ({"train_loss": [...], "val_accuracy": [...]})

    @property
    def label(self) -> str:
        # el nombre del sweep va como "<sweep>__eje=val_eje=val"; nos quedamos con la parte de ejes
        rid = self.run_id.rsplit("_", 2)[0]        # quita el timestamp _YYYYMMDD_HHMMSS
        return rid.split("__", 1)[-1] if "__" in rid else rid


def load_runs(experiments_root: str = "experiments", pattern: str | None = None) -> list[Run]:
    """Carga todas las corridas bajo `experiments_root`. `pattern` filtra por subcadena en run_id
    (p. ej. el nombre del sweep) para aislar las corridas de un barrido concreto."""
    root = Path(experiments_root)
    runs = []
    for metrics_file in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_file.parent
        if pattern and pattern not in run_dir.name:
            continue
        with open(metrics_file, encoding="utf-8") as f:
            metrics = json.load(f)
        history = {}
        hist_file = run_dir / "history.json"
        if hist_file.exists():
            with open(hist_file, encoding="utf-8") as f:
                history = json.load(f)
        runs.append(Run(run_dir.name, metrics, history))
    return runs


def plot_learning_curves(runs: list[Run], metric: str = "val_accuracy",
                         out: str | None = None):
    """Una línea por corrida: `metric` vs época. Muestra cómo aprende cada configuración."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        series = run.history.get(metric)
        if not series:
            continue
        ax.plot(range(1, len(series) + 1), series, marker="o", markersize=3, label=run.label)
    ax.set_xlabel("época")
    ax.set_ylabel(metric)
    ax.set_title(f"Curvas de aprendizaje ({metric})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    return _finish(fig, out)


def plot_pareto(runs: list[Run], x: str = "params_total", y: str = "accuracy",
                out: str | None = None):
    """Dispersión accuracy final vs nº de parámetros: coste vs rendimiento."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        xv, yv = run.metrics.get(x), run.metrics.get(y)
        if xv is None or yv is None:
            continue
        ax.scatter(xv, yv)
        ax.annotate(run.label, (xv, yv), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out)


def _finish(fig, out: str | None):
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    return fig
