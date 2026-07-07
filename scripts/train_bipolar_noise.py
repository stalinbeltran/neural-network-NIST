"""Continúa el modelo BIPOLAR (normal+invertido) añadiendo GAUSSIANO n3 al train (fine-tune).

Reanuda experiments/_inverted_mix/ckpt_inverted_mix.pt y sigue entrenando con:
  train = normal (48k) + invertido (48k) + gaussiano nivel_3 (48k) = 144k, mezclados por batch.
Monitoriza val en las TRES distribuciones (normal, invertido, gaussiano) y dibuja la curva en vivo.
Guarda en un checkpoint NUEVO (no toca el bipolar original). lr conservador (bimodal + ruido).

  python scripts/train_bipolar_noise.py --epochs 30   # 15 base + 15 nuevas
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

from nnist.data import inverted_dataset, load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed

SRC_CKPT = Path("experiments/_inverted_mix/ckpt_inverted_mix.pt")   # bipolar a reanudar (se preserva)


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return xs, ys


class LiveCurve(Callback):
    """train_loss + val_normal + val_invertido + val_gaussiano (este último arranca al reanudar)."""

    def __init__(self, val_inv_loader, val_gauss_loader, out: str):
        self.val_inv = val_inv_loader
        self.val_gauss = val_gauss_loader
        self.out = Path(out)

    def on_epoch_end(self, trainer, epoch, metrics):
        h = trainer.history
        h.setdefault("val_inverted", []).append(trainer.evaluate(self.val_inv)[0])
        h.setdefault("val_gaussiano", []).append(trainer.evaluate(self.val_gauss)[0])
        loss = h["train_loss"]; val_norm = h["val_accuracy"]
        val_inv = h["val_inverted"]; val_g = h["val_gaussiano"]
        n = len(loss)
        xs = list(range(1, n + 1))
        xs_inv = list(range(n - len(val_inv) + 1, n + 1))    # por si val_inv arranca más tarde
        xs_g = list(range(n - len(val_g) + 1, n + 1))        # gaussiano arranca al reanudar

        fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
        ax1.plot(xs, loss, marker="o", ms=3, color="tab:red", label="train_loss")
        ax1.set_xlabel("época"); ax1.set_ylabel("train_loss", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs_inv[-len(val_inv):], val_inv, marker="^", ms=3, color="tab:purple", label="val (invertido)")
        ax2.plot(xs, val_norm, marker="s", ms=3, color="tab:blue", label="val (normal)")
        ax2.plot(xs_g, val_g, marker="D", ms=3, color="tab:green", label="val (gaussiano n3)")
        ax2.set_ylabel("val_accuracy", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.axvline(len(loss) - len(val_g) + 0.5, ls=":", color="gray", lw=1)  # marca la reanudación
        ax2.set_ylim(min(min(val_norm), min(val_inv), min(val_g)) - 0.02, 1.005)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=8, ncol=2)
        ax1.set_title("Bipolar + gaussiano n3 (fine-tune) — normal / invertido / gaussiano\n"
                      f"época {n} | normal={val_norm[-1]:.4f} inv={val_inv[-1]:.4f} gauss={val_g[-1]:.4f}")
        fig.tight_layout()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out, dpi=120); plt.close(fig)
        print(f"  época {n}: loss={loss[-1]:.4f} | normal={val_norm[-1]:.4f} | "
              f"inv={val_inv[-1]:.4f} | gauss={val_g[-1]:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30, help="objetivo total (el bipolar ya hizo 15)")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tipo", default="gaussiano")
    ap.add_argument("--nivel", default="nivel_3")
    ap.add_argument("--ckpt-dir", default="experiments/_bipolar_noise")
    ap.add_argument("--out", default="reports/train_bipolar_noise.png")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed(0)
    channels = [16, 32, 64]
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=10,
                        channels=channels, dropout=args.dropout)

    clean = load_mnist()
    print(f"Datos: normal + invertido + {args.tipo} {args.nivel} (48k*3=144k)...", flush=True)
    train_mix = ConcatDataset([clean.train, inverted_dataset("train"),
                               noisy_dataset(args.tipo, args.nivel, "train")])
    train_ld = DataLoader(train_mix, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    val_norm_ld = DataLoader(clean.val, batch_size=args.batch)
    val_inv_ld = DataLoader(inverted_dataset("val"), batch_size=args.batch)
    val_gauss_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "val"), batch_size=args.batch)
    test_norm_ld = DataLoader(clean.test, batch_size=args.batch)
    test_inv_ld = DataLoader(inverted_dataset("test"), batch_size=args.batch)
    test_gauss_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "test"), batch_size=args.batch)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                      weight_decay=args.weight_decay, scheduler="cosine",
                      scheduler_params={"t_max": args.epochs})
    noise_tag = f"{args.tipo}_{args.nivel.split('_')[1]}"    # p.ej. "bajo_contraste_3"
    ckpt = Path(args.ckpt_dir) / "ckpt_bipolar_noise.pt"
    entry_id = f"bipolar_noise_{args.tipo}"
    datos = f"normal+invertido+{noise_tag}"
    trainer = Trainer(model, cfg, callbacks=[
        ModelCheckpoint(ckpt, every=1),
        LiveCurve(val_inv_ld, val_gauss_ld, args.out),
        TrainingLogger(entry_id, args.epochs, modelo=f"CNN{channels} d{args.dropout}",
                       datos=datos, checkpoint=str(ckpt)),
    ])
    info = trainer.resume_from(SRC_CKPT)   # continúa el bipolar (preserva el original)
    print(f"REANUDANDO bipolar desde época {info['epochs_done']} hasta {args.epochs} "
          f"(cosine t_max={args.epochs}, lr={args.lr})...", flush=True)
    log_training(id=entry_id, estado="en_curso", modelo=f"CNN{channels} d{args.dropout}",
                 datos=datos, épocas=f"{trainer.start_epoch}/{args.epochs}",
                 checkpoint=str(ckpt))

    trainer.fit(train_ld, val_norm_ld)

    tn = trainer.evaluate(test_norm_ld)[0]
    ti = trainer.evaluate(test_inv_ld)[0]
    tg = trainer.evaluate(test_gauss_ld)[0]
    log_training(id=entry_id, estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                 val=max(trainer.history["val_accuracy"]),
                 test=f"n {tn:.3f}/inv {ti:.3f}/g {tg:.3f}")
    print(f"\nHECHO | TEST normal={tn:.4f} | invertido={ti:.4f} | gaussiano_n3={tg:.4f} | curva en {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
