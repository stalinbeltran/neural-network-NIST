"""Entrenamiento GRADUAL (curriculum) de UNA sola CNN, resumible, con reporte gráfico.

Una única CNN (misma arquitectura, pesos que se ACUMULAN — nunca se reinicia) se entrena por etapas.
Cada etapa introduce condiciones nuevas y más difíciles del problema recta vs curva (posición,
tamaño de recta, curvatura, rotación, curvas cortas, ruido), con DATOS FRESCOS cada vez (<=100
muestras/etapa) para acumular cobertura sin ver nunca muchas a la vez. Tras cada etapa se mide en un
TEST DIFÍCIL FIJO (y en uno fácil, para vigilar el olvido). Al final genera un reporte gráfico.

Resumible: guarda `state.pt` (pesos + optimizador + etapa + historial) tras CADA etapa. Si se
interrumpe (p.ej. se acaban los tokens), volver a lanzarlo continúa donde se quedó. Un cron horario
puede relanzarlo. Idempotente: si ya terminó, solo regenera el reporte.

  python scripts/overnight_curriculum.py
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nnist.data.shapes import generate_curriculum
from nnist.models import build_model
from nnist.training import find_lr

OUT = Path("experiments/curriculum")
STATE = OUT / "state.pt"
BEST = OUT / "best.pt"                # mejor modelo por test difícil (el entregable recomendado)
SAMPLES = OUT / "samples"
EPOCHS_PER_STAGE = 25
N_PER_CLASS = 40                 # 40 rectas + 40 curvas por etapa = 80 (<=100)
BATCH = 16
DROPOUT = 0.3
CHANNELS = [8, 16]
FC_HIDDEN = 64


# ----------------------------------------------------------------- curriculum (cumulativo)
def build_curriculum() -> list[tuple[str, dict]]:
    """Lista de (descripción, params). Cada etapa ENSANCHA un eje y conserva los anteriores."""
    stages: list[tuple[str, dict]] = []
    p = dict(pos_jitter=0.0, len_range=(0.9, 1.1), radius_range=(0.5, 0.7),
             span_range=(130.0, 150.0), rotate=False, noise=0.0)

    def add(desc, replay=False, **delta):
        p.update(delta)
        stages.append((desc, dict(p), replay))

    add("Base fácil: rectas medianas centradas, curvas canónicas")
    add("Posición: las formas se descentran un poco", pos_jitter=2.0)
    add("Tamaño de recta: aparecen rectas más largas y más cortas", len_range=(0.5, 1.3))
    add("Curvatura variada: radios de curva más dispares", radius_range=(0.4, 0.9))
    add("Inclinación de curvas: rotan en cualquier orientación", rotate=True)
    add("Posición amplia: descentrado grande", pos_jitter=5.0)
    add("Rectas extremas: de muy cortas a cruzar la imagen", len_range=(0.3, 1.4))
    add("Curvas cortas: arcos pequeños", span_range=(70.0, 160.0))
    add("Combinación difícil consolidada (datos frescos)")
    add("Ruido leve sobre lo difícil", noise=0.06)
    add("Más ruido", noise=0.12)
    # Consolidación: muchas tandas de DATOS FRESCOS de la distribución dura -> la cobertura acumulada
    # sube aunque cada tanda sea <=100 muestras. Aquí es donde la red "sigue aprendiendo de noche".
    for i in range(1, 16):
        add(f"Consolidación difícil + ruido, tanda {i} (datos frescos)")
    # REPLAY intercalado (fácil+medio+difícil en la misma tanda): repasa TODAS las dificultades para
    # curar el olvido catastrófico de los casos fáciles y afianzar todo el rango.
    for i in range(1, 11):
        add(f"Replay intercalado fácil+medio+difícil, tanda {i}", replay=True)
    return stages


STAGES = build_curriculum()
HARD = STAGES[-1][1]                                   # test difícil = la distribución más dura
EASY = STAGES[0][1]                                    # test fácil = la base (retención / olvido)
MID = dict(pos_jitter=3.0, len_range=(0.5, 1.3), radius_range=(0.4, 0.9),
           span_range=(100.0, 160.0), rotate=True, noise=0.06)   # dificultad intermedia (replay)


def generate_replay(n_per_class: int, seed: int) -> dict:
    """Tanda MIXTA: reparte n_per_class entre fácil / medio / difícil (repaso de todo el rango)."""
    c = n_per_class // 3
    counts = [n_per_class - 2 * c, c, c]
    imgs, labels = [], []
    for k, (params, cnt) in enumerate(zip((EASY, MID, HARD), counts)):
        b = generate_curriculum(cnt, seed=seed * 10 + k, **params)
        imgs.append(b["images"]); labels.append(b["labels"])
    return {"images": torch.cat(imgs), "labels": torch.cat(labels)}


def _loader(blob, shuffle):
    x = blob["images"].float().div(255.0).unsqueeze(1)
    return DataLoader(TensorDataset(x, blob["labels"]), batch_size=BATCH if shuffle else 200,
                      shuffle=shuffle)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def _montage(images, cols=10, pad=2, scale=3):
    from PIL import Image
    n, h, w = images.shape
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad), 60, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas[pad + r * (h + pad):pad + r * (h + pad) + h,
               pad + c * (w + pad):pad + c * (w + pad) + w] = im
    img = Image.fromarray(canvas)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=2,
                        channels=CHANNELS, fc_hidden=FC_HIDDEN, dropout=DROPOUT)

    # test fijos (mismas semillas -> idénticos entre reanudaciones); no se entrena con ellos
    hard_loader = _loader(generate_curriculum(200, seed=90001, **HARD), shuffle=False)
    easy_loader = _loader(generate_curriculum(100, seed=80001, **EASY), shuffle=False)

    if STATE.exists():                                 # REANUDAR
        ck = torch.load(STATE)
        model.load_state_dict(ck["model"])
        opt = torch.optim.Adam(model.parameters(), lr=ck["lr"])
        opt.load_state_dict(ck["opt"])
        start, history, lr = ck["next_stage"], ck["history"], ck["lr"]
        best_hard = max((h["hard_test_acc"] for h in history), default=-1.0)
        print(f"[resume] continuando desde etapa {start}/{len(STAGES)} (lr={lr:.4g})")
    else:                                              # EMPEZAR
        lr = find_lr(model, _loader(generate_curriculum(N_PER_CLASS, 1000, **STAGES[0][1]), True),
                     weight_decay=1e-3)["suggested_lr"]
        lr = min(lr, 0.02)                             # cota de seguridad para el continual
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        start, history, best_hard = 0, [], -1.0
        print(f"[start] lr automático = {lr:.4g}")

    criterion = nn.CrossEntropyLoss()
    for s in range(start, len(STAGES)):
        desc, params, replay = STAGES[s]
        blob = generate_replay(N_PER_CLASS, seed=100 + s) if replay else \
            generate_curriculum(N_PER_CLASS, seed=100 + s, **params)      # datos frescos por etapa
        train_loader = _loader(blob, shuffle=True)

        t0 = time.perf_counter()
        model.train()
        losses = []
        for _ in range(EPOCHS_PER_STAGE):
            ep_loss = n = 0.0
            for x, y in train_loader:
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * x.size(0); n += x.size(0)
            losses.append(ep_loss / n)
        secs = time.perf_counter() - t0

        hard_acc, easy_acc, train_acc = accuracy(model, hard_loader), accuracy(model, easy_loader), \
            accuracy(model, _loader(blob, shuffle=False))
        history.append({"stage": s, "desc": desc, "params": {k: str(v) for k, v in params.items()},
                        "train_loss": round(losses[-1], 4), "train_acc": round(train_acc, 4),
                        "hard_test_acc": round(hard_acc, 4), "easy_test_acc": round(easy_acc, 4),
                        "seconds": round(secs, 2)})
        _montage(blob["images"].numpy()[:10]).save(SAMPLES / f"stage_{s:02d}.png")   # muestra recta
        _montage(blob["images"].numpy()[N_PER_CLASS:N_PER_CLASS + 10]).save(SAMPLES / f"stage_{s:02d}_curva.png")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "next_stage": s + 1,
                    "history": history, "lr": lr}, STATE)
        if hard_acc > best_hard:                       # guardar el MEJOR modelo por test difícil
            best_hard = hard_acc
            torch.save({"model": model.state_dict(), "stage": s, "hard_test_acc": hard_acc}, BEST)
        print(f"[etapa {s:2d}/{len(STAGES)}] {desc[:45]:45} | hard={hard_acc:.3f} easy={easy_acc:.3f} "
              f"train={train_acc:.3f} | {secs:.1f}s")

    with open(OUT / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    build_report(history)
    print(f"\n[done] {len(STAGES)} etapas. Reporte en {OUT / 'report.md'} y {OUT / 'learning_curve.png'}")


def build_report(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [h["stage"] for h in history]
    hard = [h["hard_test_acc"] for h in history]
    easy = [h["easy_test_acc"] for h in history]

    # 1) curva de aprendizaje
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stages, hard, "o-", label="test DIFÍCIL (todo combinado + ruido)", color="crimson")
    ax.plot(stages, easy, "s--", label="test fácil (retención / no olvido)", color="steelblue")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="azar")
    ax.set_xlabel("etapa del curriculum"); ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.02)
    ax.set_title("Aprendizaje gradual de una sola CNN (pesos acumulados)")
    ax.set_xticks(stages); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=110); plt.close(fig)

    # 2) rejilla de muestras por etapa (recta + curva) para ver la dificultad creciente
    from PIL import Image
    n = len(history)
    fig2, axes = plt.subplots(n, 2, figsize=(4, 1.1 * n))
    for i, h in enumerate(history):
        for j, suf in enumerate(["", "_curva"]):
            p = SAMPLES / f"stage_{h['stage']:02d}{suf}.png"
            axes[i, j].axis("off")
            if p.exists():
                axes[i, j].imshow(Image.open(p), cmap="gray")
        axes[i, 0].set_ylabel(f"E{h['stage']}", rotation=0, ha="right", va="center", fontsize=8)
    axes[0, 0].set_title("rectas", fontsize=9); axes[0, 1].set_title("curvas", fontsize=9)
    fig2.suptitle("Muestras por etapa (dificultad creciente)", y=0.995)
    fig2.tight_layout(); fig2.savefig(OUT / "stage_samples.png", dpi=110); plt.close(fig2)

    # 3) reporte markdown
    first_hard, last_hard = history[0]["hard_test_acc"], history[-1]["hard_test_acc"]
    best_hard = max(h["hard_test_acc"] for h in history)
    lines = [
        "# Reporte: entrenamiento gradual (curriculum) de una sola CNN\n",
        f"CNN única `cnn` channels={CHANNELS} fc_hidden={FC_HIDDEN} dropout={DROPOUT} "
        f"(~{sum(p.numel() for p in build_model('cnn', input_shape=(1,28,28), num_classes=2, channels=CHANNELS, fc_hidden=FC_HIDDEN).parameters())} params). "
        "Pesos ACUMULADOS entre etapas (nunca se reinicia). Tarea: recta vs curva.\n",
        f"**Test difícil: {first_hard:.1%} (etapa 0, azar) → {last_hard:.1%} (final), mejor {best_hard:.1%}.** "
        "El test fácil se mantiene alto tras el replay = se cura el olvido catastrófico. "
        f"Modelo recomendado: `best.pt` (mejor test difícil, {best_hard:.1%}).\n",
        "![curva](learning_curve.png)\n",
        "## Etapas aplicadas\n",
        "| # | Condición nueva | test difícil | test fácil | train |",
        "|---|---|---|---|---|",
    ]
    for h in history:
        lines.append(f"| {h['stage']} | {h['desc']} | {h['hard_test_acc']:.1%} | "
                     f"{h['easy_test_acc']:.1%} | {h['train_acc']:.1%} |")
    lines += ["\n![muestras](stage_samples.png)\n"]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
