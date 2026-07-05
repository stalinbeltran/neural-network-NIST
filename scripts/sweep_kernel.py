"""Sweep del TAMAÑO DE KERNEL (3, 5, 7) sobre la misma CNN regularizada [16,32,64]+dropout,
entrenada en LIMPIO (val/test también limpios). Mismo régimen que la última corrida:
weight_decay=1e-4, dropout=0.2, scheduler=cosine.

Entrena en ROUND-ROBIN por tramos: cada modelo avanza `--chunk` épocas por vuelta, luego el
siguiente, hasta que los tres llegan a `--epochs`. Así las 3 curvas crecen a la vez y el reporte
gráfico (reports/kernel_sweep.png) muestra el avance progresivo, actualizándose tras cada época.

Reanudable: cada kernel guarda su checkpoint en experiments/_kernel_sweep/ckpt_k<k>.pt.

  python scripts/sweep_kernel.py                       # 3,5,7 hasta 15 épocas, tramos de 3
  python scripts/sweep_kernel.py --epochs 15 --resume  # reanudar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from nnist.data import load_mnist
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed

CKPT_DIR = Path("experiments") / "_kernel_sweep"


class PrintEpoch(Callback):
    """Imprime val (limpio) al terminar cada época de un kernel dado."""
    def __init__(self, k: int):
        self.k = k

    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"  [k={self.k}] época {epoch + 1}: val_limpio={metrics['val_accuracy']:.4f}", flush=True)


class LivePlot(Callback):
    """Regenera reports/kernel_sweep.png tras CADA época, leyendo la historia en vivo de los 3 kernels."""
    def __init__(self, trainers: dict, kernels: list):
        self.trainers = trainers
        self.kernels = kernels

    def on_epoch_end(self, trainer, epoch, metrics):
        plot_history({k: self.trainers[k].history["val_accuracy"] for k in self.kernels})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15, help="épocas totales por modelo")
    ap.add_argument("--chunk", type=int, default=3, help="épocas que avanza cada modelo por vuelta")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--resume", action="store_true", help="reanudar desde los checkpoints existentes")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    channels = [16, 32, 64]
    input_shape = (1, 28, 28)

    print("Preparando datos (MNIST limpio)...", flush=True)
    clean = load_mnist()
    train_ld = DataLoader(clean.train, batch_size=args.batch, shuffle=True)
    val_ld = DataLoader(clean.val, batch_size=args.batch)
    test_ld = DataLoader(clean.test, batch_size=args.batch)

    def make_cfg():
        return TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                           weight_decay=args.weight_decay, scheduler="cosine",
                           scheduler_params={"t_max": args.epochs})

    trainers: dict = {}
    for k in args.kernels:
        set_seed(args.seed)   # misma semilla de init (dentro de lo posible; el kernel cambia la forma)
        model = build_model("cnn", input_shape=input_shape, num_classes=10, channels=channels,
                            kernel_size=k, dropout=args.dropout)
        ckpt = CKPT_DIR / f"ckpt_k{k}.pt"
        entry_id = f"kernel_k{k}"
        logger = TrainingLogger(entry_id, args.epochs, modelo=f"CNN{channels} k={k} d{args.dropout}",
                                datos="mnist_limpio", checkpoint=str(ckpt))
        trainers[k] = Trainer(model, make_cfg(),
                              callbacks=[ModelCheckpoint(ckpt, every=1), PrintEpoch(k), logger])
        # registra la corrida como EN CURSO desde el arranque (aunque aún no haya épocas)
        log_training(id=entry_id, estado="en_curso", modelo=f"CNN{channels} k={k} d{args.dropout}",
                     datos="mnist_limpio", épocas=f"0/{args.epochs}", checkpoint=str(ckpt))

    live = LivePlot(trainers, args.kernels)
    for k in args.kernels:
        trainers[k].callbacks.append(live)

    if args.resume:
        for k in args.kernels:
            info = trainers[k].resume_from(CKPT_DIR / f"ckpt_k{k}.pt")
            print(f"REANUDANDO k={k} desde época {info['epochs_done']}", flush=True)

    for k in args.kernels:
        p = trainers[k].model.count_params()["params_total"]
        print(f"Modelo k={k}: CNN {channels} kernel={k} dropout={args.dropout} "
              f"wd={args.weight_decay} scheduler=cosine | params={p:,}", flush=True)

    # ROUND-ROBIN por tramos: k=3 avanza `chunk` épocas, luego k=5, luego k=7, y otra vuelta...
    target = 0
    while target < args.epochs:
        target = min(target + args.chunk, args.epochs)
        for k in args.kernels:
            trainers[k].cfg.epochs = target
            print(f"-> entrenando k={k} hasta época {target}", flush=True)
            trainers[k].fit(train_ld, val_ld)

    # tabla comparativa por época (val limpio)
    header = f"{'época':<7}" + "".join(f"{'k=' + str(k):<11}" for k in args.kernels)
    print(f"\n{header}", flush=True)
    n = max(len(trainers[k].history["val_accuracy"]) for k in args.kernels)
    for i in range(n):
        row = f"{i + 1:<7}"
        for k in args.kernels:
            h = trainers[k].history["val_accuracy"]
            row += f"{h[i]:<11.4f}" if i < len(h) else " " * 11
        print(row, flush=True)

    # evaluación FINAL en TEST (limpio) + persistencia
    results = {}
    print("", flush=True)
    for k in args.kernels:
        acc = trainers[k].evaluate(test_ld)[0]
        params = trainers[k].model.count_params()["params_total"]
        val_final = trainers[k].history["val_accuracy"][-1]
        results[str(k)] = {"test_clean": acc, "params": params,
                           "val": trainers[k].history["val_accuracy"]}
        print(f"TEST (limpio) k={k}: acc={acc:.4f} | params={params:,}", flush=True)
        # marca la corrida como HECHA con el test final en la bitácora
        log_training(id=f"kernel_k{k}", estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                     val=val_final, test=acc)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / "history.json").write_text(json.dumps(
        {"kernels": args.kernels, "channels": channels, "dropout": args.dropout,
         "weight_decay": args.weight_decay, "results": results}, indent=2), encoding="utf-8")

    plot_history({k: results[str(k)]["val"] for k in args.kernels})


def plot_history(hist_by_k: dict, out: str = "reports/kernel_sweep.png") -> None:
    """Gráfica de val (limpio) por época: una curva por tamaño de kernel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(sin gráfica: {e})", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, h in hist_by_k.items():
        ax.plot(range(1, len(h) + 1), h, "o-", label=f"kernel {k}")
    ax.set_xlabel("época"); ax.set_ylabel("val_accuracy (evaluado en LIMPIO)")
    ax.set_title("Sweep de kernel_size — CNN [16,32,64]+dropout (datos limpios)")
    ax.grid(True, alpha=0.3); ax.legend()
    n = max((len(h) for h in hist_by_k.values()), default=0)
    if 0 < n <= 15:
        ax.set_xticks(list(range(1, n + 1)))
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)   # LivePlot llama esto cada época; cerrar evita acumular figuras en memoria
    print(f"Gráfica: {out}", flush=True)


if __name__ == "__main__":
    main()
