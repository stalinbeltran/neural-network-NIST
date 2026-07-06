"""Entrena UNA CNN sobre NORMAL + INVERTIDO simultáneamente (ambas polaridades), curva en vivo.

Train = unión de MNIST normal (48k) + su versión invertida (48k) = 96k, mezclados por batch.
Monitoriza val en las DOS polaridades para ver si recupera ambas. Curva en vivo tras cada época
en reports/train_inverted_mix.png. Registra en la bitácora y guarda checkpoint reanudable.

  python scripts/train_inverted_mix.py --epochs 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader

from nnist.data import inverted_dataset, load_mnist
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed


def _collate(batch):
    """Normaliza etiquetas a tensor: el train mezcla clean.train (label int) e invertido (label tensor)."""
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return xs, ys


class LiveCurve(Callback):
    """Redibuja la curva tras cada época: train_loss + val_normal + val_invertido."""

    def __init__(self, val_inv_loader, out: str):
        self.val_inv = val_inv_loader
        self.out = Path(out)

    def on_epoch_end(self, trainer, epoch, metrics):
        val_inv = trainer.history.setdefault("val_inverted", [])   # persiste en el checkpoint
        val_inv.append(trainer.evaluate(self.val_inv)[0])
        loss = trainer.history["train_loss"]
        val_norm = trainer.history["val_accuracy"]
        xs = list(range(1, len(loss) + 1))

        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(xs, loss, marker="o", color="tab:red", label="train_loss")
        ax1.set_xlabel("época"); ax1.set_ylabel("train_loss", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.set_xticks(xs); ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs, val_norm, marker="s", color="tab:blue", label="val_acc (normal)")
        ax2.plot(xs, val_inv, marker="^", color="tab:purple", label="val_acc (invertido)")
        ax2.set_ylabel("val_accuracy", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_ylim(min(min(val_norm), min(val_inv)) - 0.02, 1.005)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
        ax1.set_title("Aprendizaje — CNN entrenada con NORMAL + INVERTIDO\n"
                      f"época {len(xs)} | val_normal={val_norm[-1]:.4f} | val_invertido={val_inv[-1]:.4f}")
        fig.tight_layout()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out, dpi=120); plt.close(fig)
        print(f"  época {len(xs)}: loss={loss[-1]:.4f} | val_normal={val_norm[-1]:.4f} | "
              f"val_invertido={val_inv[-1]:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="experiments/_inverted_mix")
    ap.add_argument("--out", default="reports/train_inverted_mix.png")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed(args.seed)
    channels = [16, 32, 64]
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=10,
                        channels=channels, dropout=args.dropout)

    clean = load_mnist()
    print("Preparando datos: normal + invertido (train 48k+48k=96k)...", flush=True)
    train_mix = ConcatDataset([clean.train, inverted_dataset("train")])
    train_ld = DataLoader(train_mix, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    val_norm_ld = DataLoader(clean.val, batch_size=args.batch)          # val normal (primario)
    val_inv_ld = DataLoader(inverted_dataset("val"), batch_size=args.batch)
    test_norm_ld = DataLoader(clean.test, batch_size=args.batch)
    test_inv_ld = DataLoader(inverted_dataset("test"), batch_size=args.batch)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                      weight_decay=args.weight_decay, scheduler="cosine",
                      scheduler_params={"t_max": args.epochs})
    ckpt = Path(args.ckpt_dir) / "ckpt_inverted_mix.pt"
    entry_id = "inverted_mix"
    trainer = Trainer(model, cfg, callbacks=[
        ModelCheckpoint(ckpt, every=1),
        LiveCurve(val_inv_ld, args.out),
        TrainingLogger(entry_id, args.epochs, modelo=f"CNN{channels} d{args.dropout}",
                       datos="normal+invertido", checkpoint=str(ckpt)),
    ])
    log_training(id=entry_id, estado="en_curso", modelo=f"CNN{channels} d{args.dropout}",
                 datos="normal+invertido", épocas=f"0/{args.epochs}", checkpoint=str(ckpt))

    print(f"Entrenando {args.epochs} épocas sobre normal+invertido...", flush=True)
    trainer.fit(train_ld, val_norm_ld)

    test_norm = trainer.evaluate(test_norm_ld)[0]
    test_inv = trainer.evaluate(test_inv_ld)[0]
    val_best = max(trainer.history["val_accuracy"])
    log_training(id=entry_id, estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                 val=val_best, test=f"norm {test_norm:.4f} / inv {test_inv:.4f}")
    print(f"\nHECHO | TEST normal={test_norm:.4f} | TEST invertido={test_inv:.4f} | curva en {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
