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
import numpy as np  # noqa: E402


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

    @property
    def group(self) -> str:
        """Configuración OFAT sin la semilla (varias corridas comparten group)."""
        return self.label.split("__seed", 1)[0]


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


def plot_ofat_curves(runs: list[Run], metric: str = "val_accuracy", out: str | None = None):
    """Curvas de aprendizaje agrupadas por configuración OFAT: media de las semillas ± desviación.

    Una línea (con banda) por `group`, promediando las corridas que solo difieren en la semilla.
    Muestra el efecto de cada hiperparámetro con su variabilidad entre inicializaciones.
    """
    groups: dict[str, list[list[float]]] = {}
    for run in runs:
        series = run.history.get(metric)
        if series:
            groups.setdefault(run.group, []).append(series)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for group, seed_series in sorted(groups.items()):
        n = min(len(s) for s in seed_series)          # recorta a la longitud común de época
        arr = np.array([s[:n] for s in seed_series])  # (n_seeds, n_epochs)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        epochs = range(1, n + 1)
        line, = ax.plot(epochs, mean, marker="o", markersize=3,
                        label=f"{group} (n={len(seed_series)})")
        if arr.shape[0] > 1:
            ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("época")
    ax.set_ylabel(f"{metric} (media ± std entre semillas)")
    ax.set_title("OFAT: efecto de cada hiperparámetro en el aprendizaje")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    return _finish(fig, out)


def plot_comparison(datasets: dict[str, list[dict]], metric: str = "test_mean",
                    std_key: str | None = None, out: str | None = None):
    """Compara varias baterías (p. ej. MLP vs CNN) en accuracy vs nº de parámetros.

    `datasets`: {etiqueta: filas del summary CSV}. Cada fila necesita `group`, `params_total`
    y la columna `metric`. Params en escala log (abarcan de ~50k a ~800k).
    """
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for label, rows in datasets.items():
        xs = [float(r["params_total"]) for r in rows]
        ys = [float(r[metric]) for r in rows]
        errs = [float(r.get(std_key, 0) or 0) for r in rows] if std_key else None
        ax.errorbar(xs, ys, yerr=errs, fmt="o", capsize=3, alpha=0.85, label=label)
        for r in rows:
            ax.annotate(r["group"], (float(r["params_total"]), float(r[metric])),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("nº de parámetros (escala log)")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparación por configuración: {metric} vs parámetros")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    return _finish(fig, out)


def plot_axis_effects(datasets: dict[str, list[dict]], metric: str = "test_mean",
                      out: str | None = None):
    """Efecto de cada eje OFAT por modelo: Δ = variante − baseline (en puntos %).

    Una sub-gráfica por eje; barras por variante, agrupadas por modelo. Positivo = mejora sobre
    el baseline del modelo; negativo = empeora. Los ejes compartidos (lr/batch/dropout) muestran
    ambos modelos; los propios de una arquitectura, solo ese modelo. Medir Δ respecto al baseline
    de CADA modelo neutraliza su nivel de partida y compara la SENSIBILIDAD a cada eje.
    """
    per_model: dict[str, dict[str, dict[str, float]]] = {}
    for model, rows in datasets.items():
        base = next((r for r in rows if r["group"] == "baseline"), None)
        if base is None:
            continue
        b = float(base[metric])
        axes: dict[str, dict[str, float]] = {}
        for r in rows:
            axis, sep, val = r["group"].partition("=")
            if not sep:            # baseline u otros sin '='
                continue
            axes.setdefault(axis, {})[val] = (float(r[metric]) - b) * 100.0
        per_model[model] = axes

    models = list(per_model.keys())
    ordered: list[str] = []
    for ax_name in ("lr", "batch_size", "dropout"):     # compartidos primero
        if any(ax_name in per_model[m] for m in models):
            ordered.append(ax_name)
    for m in models:                                    # luego los propios de cada modelo
        for ax_name in per_model[m]:
            if ax_name not in ordered:
                ordered.append(ax_name)

    def sort_key(v: str):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    ncols = 3
    nrows = (len(ordered) + ncols - 1) // ncols
    fig, axarr = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ax_name in enumerate(ordered):
        ax = axarr[i // ncols][i % ncols]
        variants = sorted({v for m in models for v in per_model[m].get(ax_name, {})}, key=sort_key)
        x = np.arange(len(variants))
        width = 0.8 / max(len(models), 1)
        for j, model in enumerate(models):
            vals = [per_model[model].get(ax_name, {}).get(v, np.nan) for v in variants]
            ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width,
                   label=model, color=colors[j % len(colors)])
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(ax_name)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=7, rotation=30)
        ax.set_ylabel("Δ vs baseline (pp)", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    for k in range(len(ordered), nrows * ncols):        # ocultar celdas sobrantes
        axarr[k // ncols][k % ncols].axis("off")

    handles, labels = axarr[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"Efecto de cada eje por modelo (Δ {metric} vs baseline, en puntos %)")
    return _finish(fig, out)


def _finish(fig, out: str | None):
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    return fig
