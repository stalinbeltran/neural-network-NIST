"""Sweep del TIPO DE RUIDO en el entrenamiento sobre UNA sola CNN [16,32,64]+dropout.

Idea: ¿qué tipo de ruido, usado como augmentation al ENTRENAR, mejora el aprendizaje? Se entrena
la misma red por separado con cada tipo de ruido (a un nivel fijo) y con datos LIMPIOS (baseline);
VAL y TEST son SIEMPRE LIMPIOS. Un tipo "mejora el aprendizaje" si su accuracy limpia supera a la
del baseline limpio (el ruido actúa como regularización/augmentation).

Pensado para acabar en ~media hora (pocas épocas). Genera on-demand los sets ruidosos de train y
los cachea. Reporta el avance progresivamente en reports/noise_type_sweep.png (barras, val limpio
por tipo) y registra cada corrida en la bitácora trainings/TRAININGS.md.

  python scripts/sweep_noise_type.py --epochs 3 --nivel nivel_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from nnist.data import load_levels, load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed

CKPT_DIR = Path("experiments") / "_noisetype_sweep"
BASELINE = "limpio"


class PrintEpoch(Callback):
    def __init__(self, label: str):
        self.label = label

    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"  [{self.label:<20}] época {epoch + 1}: val_limpio={metrics['val_accuracy']:.4f}", flush=True)


class LivePlot(Callback):
    """Regenera el gráfico de barras (val limpio por tipo de ruido) tras cada época."""
    def __init__(self, trainers: dict, labels: list, nivel: str, out: str):
        self.trainers = trainers
        self.labels = labels
        self.nivel = nivel
        self.out = out

    def on_epoch_end(self, trainer, epoch, metrics):
        scores = {}
        for lb in self.labels:
            h = self.trainers[lb].history["val_accuracy"]
            scores[lb] = max(h) if h else 0.0   # mejor val alcanzada hasta ahora
        plot_bars(scores, self.nivel, out=self.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--nivel", default="nivel_2", help="nivel de ruido fijo para todos los tipos")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR),
                    help="carpeta de checkpoints (usar una copia para no tocar corridas en prueba)")
    ap.add_argument("--id-prefix", default="noisetype", help="prefijo del id en la bitácora")
    ap.add_argument("--plot-out", default="reports/noise_type_sweep.png")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    channels = [16, 32, 64]
    input_shape = (1, 28, 28)
    ckpt_dir = Path(args.ckpt_dir)                 # configurable: usar copia para no tocar originales
    tipos = list(load_levels()["types"])          # los 11 tipos, en orden del YAML
    labels = [BASELINE] + tipos                    # baseline limpio primero

    print(f"Preparando datos... tipos={len(tipos)} | nivel={args.nivel} | épocas={args.epochs}", flush=True)
    clean = load_mnist()
    val_ld = DataLoader(clean.val, batch_size=args.batch)     # VAL limpio
    test_ld = DataLoader(clean.test, batch_size=args.batch)   # TEST limpio

    def make_cfg():
        return TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                           weight_decay=args.weight_decay, scheduler="cosine",
                           scheduler_params={"t_max": args.epochs})

    # loaders de train: limpio para el baseline, ruidoso (por tipo) para el resto
    def train_loader(label: str) -> DataLoader:
        if label == BASELINE:
            return DataLoader(clean.train, batch_size=args.batch, shuffle=True)
        return DataLoader(noisy_dataset(label, args.nivel, "train"), batch_size=args.batch, shuffle=True)

    trainers: dict = {}
    for lb in labels:
        set_seed(args.seed)   # misma init de pesos para todos (comparación justa)
        model = build_model("cnn", input_shape=input_shape, num_classes=10, channels=channels,
                            dropout=args.dropout)
        ckpt = ckpt_dir / f"ckpt_{lb}.pt"
        entry_id = f"{args.id_prefix}_{lb}"
        datos = "mnist_limpio" if lb == BASELINE else f"{lb} {args.nivel}"
        logger = TrainingLogger(entry_id, args.epochs, modelo=f"CNN{channels} d{args.dropout}",
                                datos=datos, checkpoint=str(ckpt))
        trainers[lb] = Trainer(model, make_cfg(),
                               callbacks=[ModelCheckpoint(ckpt, every=1), PrintEpoch(lb), logger])
        log_training(id=entry_id, estado="en_curso", modelo=f"CNN{channels} d{args.dropout}",
                     datos=datos, épocas=f"0/{args.epochs}", checkpoint=str(ckpt))

    live = LivePlot(trainers, labels, args.nivel, args.plot_out)
    for lb in labels:
        trainers[lb].callbacks.append(live)

    if args.resume:
        for lb in labels:
            info = trainers[lb].resume_from(ckpt_dir / f"ckpt_{lb}.pt")
            print(f"REANUDANDO {lb} desde época {info['epochs_done']} (hasta {args.epochs})", flush=True)

    # entrena cada modelo (secuencial); el baseline primero como referencia
    results = {}
    for lb in labels:
        print(f"\n=== Entrenando '{lb}' ===", flush=True)
        trainers[lb].fit(train_loader(lb), val_ld)
        acc = trainers[lb].evaluate(test_ld)[0]
        val_best = max(trainers[lb].history["val_accuracy"])
        results[lb] = {"test_clean": acc, "val_best": val_best,
                       "val": trainers[lb].history["val_accuracy"]}
        log_training(id=f"{args.id_prefix}_{lb}", estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                     val=val_best, test=acc)
        print(f"  -> {lb}: TEST_limpio={acc:.4f} | mejor val={val_best:.4f}", flush=True)

    # ranking final vs baseline
    base_test = results[BASELINE]["test_clean"]
    print(f"\n{'tipo de ruido':<22}{'test_limpio':<13}{'Δ vs limpio (pp)':<18}", flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["test_clean"], reverse=True)
    for lb, r in ranked:
        delta = (r["test_clean"] - base_test) * 100
        marca = "  <- baseline" if lb == BASELINE else (" (mejora)" if delta > 0 else "")
        print(f"{lb:<22}{r['test_clean']:<13.4f}{delta:<+18.2f}{marca}", flush=True)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "history.json").write_text(json.dumps(
        {"nivel": args.nivel, "epochs": args.epochs, "channels": channels,
         "baseline_test": base_test, "results": results}, indent=2), encoding="utf-8")

    plot_bars({lb: results[lb]["val_best"] for lb in labels}, args.nivel, out=args.plot_out)


def plot_bars(scores: dict, nivel: str, out: str = "reports/noise_type_sweep.png") -> None:
    """Barras horizontales: mejor val (limpio) por tipo de ruido de entrenamiento; línea = baseline limpio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(sin gráfica: {e})", flush=True)
        return
    base = scores.get(BASELINE, 0.0)
    # ordena de mayor a menor (dejando fuera el baseline, que se marca con una línea)
    items = sorted(((lb, v) for lb, v in scores.items() if lb != BASELINE),
                   key=lambda kv: kv[1])
    ys = [lb for lb, _ in items]
    xs = [v for _, v in items]
    colors = ["tab:green" if v > base else "tab:red" for v in xs]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ys, xs, color=colors)
    if base > 0:
        ax.axvline(base, color="black", linestyle="--", label=f"baseline limpio ({base:.4f})")
        ax.legend(loc="lower right")
    ax.set_xlabel("mejor val_accuracy (evaluado en LIMPIO)")
    ax.set_title(f"¿Qué ruido de entrenamiento ayuda? — CNN [16,32,64], nivel={nivel}\n"
                 "verde = supera al baseline limpio, rojo = por debajo")
    if xs:
        lo = min(min(xs), base) - 0.01
        ax.set_xlim(max(0.0, lo), 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Gráfica: {out}", flush=True)


if __name__ == "__main__":
    main()
