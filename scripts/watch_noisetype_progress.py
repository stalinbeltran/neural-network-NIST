"""Watcher de PROGRESO del sweep de tipos de ruido multi-semilla.

Lee la bitácora (trainings/TRAININGS.md) cada `--every` s y re-dibuja
reports/noisetype_seeds_progress.png: por tipo de entrenamiento, la accuracy en TEST limpio
de cada semilla ya terminada + media±std acumulada. Se refresca solo (abre el PNG en VSCode y
se actualiza al cambiar). Termina cuando las 9 semillas × 12 modelos están 'hecho', o al agotar
`--max-min`.

Uso:  python scripts/watch_noisetype_progress.py --seeds 9 --every 30
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nnist.data import load_levels
from nnist.utils.trainlog import LOG_PATH, _parse

OUT = Path("reports/noisetype_seeds_progress.png")
ID_RE = re.compile(r"^noisetype_(?:s(\d+)_)?(.+)$")


def collect(n_seeds: int):
    """Devuelve {tipo: {seed: test_float}} y (n_hecho, n_total) leyendo la bitácora."""
    types = ["limpio"] + list(load_levels()["types"])
    data = {t: {} for t in types}
    n_done = 0
    try:
        rows, _ = _parse(Path(LOG_PATH))
    except Exception:
        return data, 0, len(types) * n_seeds, types
    for rid, row in rows.items():
        m = ID_RE.match(rid)
        if not m:
            continue
        seed = int(m.group(1)) if m.group(1) else 0
        tipo = m.group(2)
        if tipo not in data or seed >= n_seeds:
            continue
        if row.get("estado") == "hecho":
            n_done += 1
            try:
                data[tipo][seed] = float(row.get("test", "").replace(",", "."))
            except (ValueError, AttributeError):
                pass
    return data, n_done, len(types) * n_seeds, types


def render(data, n_done, n_total, types, n_seeds):
    import numpy as np
    base_vals = list(data["limpio"].values())
    base = float(np.mean(base_vals)) if base_vals else None

    order = [t for t in types if t != "limpio"]
    order = sorted(order, key=lambda t: (np.mean(list(data[t].values())) if data[t] else 0))
    order = ["limpio"] + order   # limpio arriba

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, t in enumerate(order):
        vals = list(data[t].values())
        if vals:
            mean = float(np.mean(vals)); std = float(np.std(vals))
            color = "tab:gray" if t == "limpio" else ("tab:green" if base and mean > base else "tab:red")
            ax.barh(i, mean, color=color, alpha=0.55,
                    xerr=std if len(vals) > 1 else None, capsize=3,
                    error_kw={"ecolor": "black", "lw": 1})
            ax.scatter(vals, [i] * len(vals), color="black", s=14, zorder=3)
            ax.text(mean + 0.002, i, f"{mean:.4f} (n={len(vals)})", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    if base:
        ax.axvline(base, ls="--", color="black", lw=1, label=f"baseline limpio ({base:.4f})")
        ax.legend(loc="lower left", fontsize=8)
    lo = min([v for d in data.values() for v in d.values()] or [0.9]) - 0.01
    ax.set_xlim(max(0.0, lo), 1.0)
    ax.set_xlabel("accuracy en TEST limpio (puntos = semillas; barra = media ± std)")
    ax.set_title(f"Sweep tipo de ruido — progreso {n_done}/{n_total} modelos "
                 f"({n_seeds} semillas × 12)\nactualizado {time.strftime('%H:%M:%S')}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=9, help="nº total de semillas esperadas (0..seeds-1)")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--max-min", type=float, default=300.0)
    args = ap.parse_args()
    try:
        import sys; sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    while True:
        data, n_done, n_total, types = collect(args.seeds)
        render(data, n_done, n_total, types, args.seeds)
        print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{n_total} modelos hechos -> {OUT}", flush=True)
        if n_done >= n_total:
            print("COMPLETADO: todas las semillas terminadas.", flush=True)
            break
        if (time.time() - t0) / 60 >= args.max_min:
            print("Fin del watcher (max-min alcanzado).", flush=True)
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
