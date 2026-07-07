"""Entrena UNA CNN sobre los DOS sets ruidosos con polaridad opuesta (gaussiano nivel_3):

  train = invertido+gaussiano_n3 (48k)  +  normal+gaussiano_n3 (48k)  = 96k, mezclados por batch.

Ambos streams están degradados con el MISMO tipo/nivel de ruido; lo único que cambia es la
polaridad fondo/trazo. Monitoriza val en las DOS distribuciones (normal-ruidoso e invertido-ruidoso)
para ver si la red aprende a clasificar en ambas a la vez pese al ruido.

Entrenamiento GRADUAL y REANUDABLE: `ModelCheckpoint(every=1)` guarda el estado (pesos + optimizador
+ época + historial) TRAS CADA ÉPOCA en un único checkpoint. Puedes detener el proceso (Ctrl-C) en
cualquier momento y evaluar/continuar el estado guardado. Para continuar más épocas:

  python scripts/train_inverted_noisy_mix.py --epochs 40 --resume    # sube el objetivo y sigue

Primera corrida (arranca de cero):
  python scripts/train_inverted_noisy_mix.py --epochs 30
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

from nnist.data import inverted_noisy_dataset, noisy_dataset
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed


def _collate(batch):
    """Etiquetas a tensor long (ambos streams son TensorDataset, pero unificamos por robustez)."""
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return xs, ys


class LiveCurve(Callback):
    """Redibuja tras cada época: train_loss + val (normal-ruidoso) + val (invertido-ruidoso)."""

    def __init__(self, val_norm_loader, val_inv_loader, out: str):
        # val_norm_loader es el val PRIMARIO que el Trainer ya evalúa (-> history['val_accuracy']);
        # aquí solo añadimos el invertido-ruidoso como serie extra.
        self.val_inv = val_inv_loader
        self.out = Path(out)

    def on_epoch_end(self, trainer, epoch, metrics):
        h = trainer.history
        vi = h.setdefault("val_inverted_noisy", [])
        vi.append(trainer.evaluate(self.val_inv)[0])
        loss = h["train_loss"]; val_norm = h["val_accuracy"]
        xs = list(range(1, len(loss) + 1))

        fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
        ax1.plot(xs, loss, marker="o", ms=3, color="tab:red", label="train_loss")
        ax1.set_xlabel("época"); ax1.set_ylabel("train_loss", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs, val_norm, marker="s", ms=3, color="tab:blue", label="val (normal+gauss n3)")
        ax2.plot(xs, vi, marker="^", ms=3, color="tab:purple", label="val (invertido+gauss n3)")
        ax2.set_ylabel("val_accuracy", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_ylim(min(min(val_norm), min(vi)) - 0.02, 1.005)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=8, ncol=2)
        ax1.set_title("CNN sobre invertido+ruido y normal+ruido (gaussiano n3)\n"
                      f"época {len(xs)} | val_normal={val_norm[-1]:.4f} | val_invertido={vi[-1]:.4f}")
        fig.tight_layout()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out, dpi=120); plt.close(fig)
        print(f"  época {len(xs)}: loss={loss[-1]:.4f} | val_normal={val_norm[-1]:.4f} | "
              f"val_invertido={vi[-1]:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tipo", default="gaussiano")
    ap.add_argument("--nivel", default="nivel_3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="experiments/_inverted_noisy_mix")
    ap.add_argument("--out", default="reports/train_inverted_noisy_mix.png")
    ap.add_argument("--resume", action="store_true", help="reanuda el checkpoint si existe")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed(args.seed)
    channels = [16, 32, 64]
    model = build_model("cnn", input_shape=(1, 28, 28), num_classes=10,
                        channels=channels, dropout=args.dropout)

    noise_tag = f"{args.tipo}_{args.nivel.split('_')[1]}"      # p.ej. "gaussiano_3"
    print(f"Datos: invertido+{noise_tag} (48k) + normal+{noise_tag} (48k) = 96k...", flush=True)
    train_mix = ConcatDataset([
        inverted_noisy_dataset(args.tipo, args.nivel, "train"),
        noisy_dataset(args.tipo, args.nivel, "train"),
    ])
    train_ld = DataLoader(train_mix, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    val_norm_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "val"), batch_size=args.batch)
    val_inv_ld = DataLoader(inverted_noisy_dataset(args.tipo, args.nivel, "val"), batch_size=args.batch)
    test_norm_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "test"), batch_size=args.batch)
    test_inv_ld = DataLoader(inverted_noisy_dataset(args.tipo, args.nivel, "test"), batch_size=args.batch)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                      weight_decay=args.weight_decay, scheduler="cosine",
                      scheduler_params={"t_max": args.epochs})
    ckpt = Path(args.ckpt_dir) / "ckpt_inverted_noisy_mix.pt"
    entry_id = f"inverted_noisy_mix_{args.tipo}"
    datos = f"invertido+{noise_tag} & normal+{noise_tag}"
    trainer = Trainer(model, cfg, callbacks=[
        ModelCheckpoint(ckpt, every=1),                       # guarda TRAS CADA ÉPOCA (reanudable)
        LiveCurve(val_norm_ld, val_inv_ld, args.out),
        TrainingLogger(entry_id, args.epochs, modelo=f"CNN{channels} d{args.dropout}",
                       datos=datos, checkpoint=str(ckpt)),
    ])
    if args.resume and ckpt.exists():
        info = trainer.resume_from(ckpt)
        print(f"REANUDANDO desde época {info['epochs_done']} hasta {args.epochs}...", flush=True)
    log_training(id=entry_id, estado="en_curso", modelo=f"CNN{channels} d{args.dropout}",
                 datos=datos, épocas=f"{trainer.start_epoch}/{args.epochs}", checkpoint=str(ckpt))

    print(f"Modelo: CNN{channels} | params={model.count_params()['params_total']:,} | "
          f"cosine t_max={args.epochs} lr={args.lr} wd={args.weight_decay}\n", flush=True)
    trainer.fit(train_ld, val_norm_ld)

    tn = trainer.evaluate(test_norm_ld)[0]
    ti = trainer.evaluate(test_inv_ld)[0]
    log_training(id=entry_id, estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                 val=max(trainer.history["val_accuracy"]),
                 test=f"n {tn:.3f}/inv {ti:.3f}")
    print(f"\nHECHO | TEST normal+{noise_tag}={tn:.4f} | invertido+{noise_tag}={ti:.4f} | "
          f"checkpoint={ckpt} | curva={args.out}", flush=True)


if __name__ == "__main__":
    main()
