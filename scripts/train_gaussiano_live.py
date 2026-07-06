"""Entrena UNA CNN con ruido gaussiano nivel_3 (sigma=0.5) y dibuja la curva de aprendizaje EN VIVO.

Tras CADA época redibuja reports/train_gaussiano_live.png con:
  - train_loss (eje izq.)   -> cómo baja la pérdida sobre el train ruidoso
  - val_accuracy RUIDOSO    -> aprendizaje de la propia tarea (val con el mismo ruido)
  - val_accuracy LIMPIO     -> generalización a imágenes sin ruido

Abre el PNG en VSCode para verlo refrescarse solo. Registra en la bitácora y guarda checkpoint
reanudable. Datos ruidosos on-demand (se cachean).

Uso:  python scripts/train_gaussiano_live.py --epochs 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nnist.data import load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed


class LiveCurve(Callback):
    """Redibuja la curva de aprendizaje tras cada época (train_loss + val ruidoso + val limpio)."""

    def __init__(self, clean_val_loader, out: str, tipo: str, nivel: str):
        self.clean_val = clean_val_loader
        self.out = Path(out)
        self.tipo, self.nivel = tipo, nivel

    def on_epoch_end(self, trainer, epoch, metrics):
        # val limpio se guarda en el historial del trainer -> persiste en el checkpoint y sobrevive al resume
        val_clean = trainer.history.setdefault("val_clean", [])
        acc_clean = trainer.evaluate(self.clean_val)[0]
        val_clean.append(acc_clean)
        loss = trainer.history["train_loss"]
        val_noisy = trainer.history["val_accuracy"]
        xs = list(range(1, len(loss) + 1))

        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(xs, loss, marker="o", color="tab:red", label="train_loss (ruidoso)")
        ax1.set_xlabel("época")
        ax1.set_ylabel("train_loss", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax1.set_xticks(xs)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs, val_noisy, marker="s", color="tab:blue", label="val_acc (ruidoso)")
        ax2.plot(xs, val_clean, marker="^", color="tab:green", label="val_acc (limpio)")
        ax2.set_ylabel("val_accuracy", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_ylim(min(min(val_noisy), min(val_clean)) - 0.02, 1.005)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
        ax1.set_title(f"Aprendizaje — CNN con ruido {self.tipo} {self.nivel} (sigma=0.5)\n"
                      f"época {len(xs)} | val_ruidoso={val_noisy[-1]:.4f} | val_limpio={acc_clean:.4f}")
        fig.tight_layout()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out, dpi=120)
        plt.close(fig)
        print(f"  época {len(xs)}: loss={loss[-1]:.4f} | val_ruidoso={val_noisy[-1]:.4f} | "
              f"val_limpio={acc_clean:.4f} -> {self.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--tipo", default="gaussiano")
    ap.add_argument("--nivel", default="nivel_3")
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="experiments/_gaussiano_live")
    ap.add_argument("--out", default="reports/train_gaussiano_live.png")
    ap.add_argument("--resume", action="store_true",
                    help="reanuda el checkpoint existente y continúa hasta --epochs (t_max=--epochs).")
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
    print(f"Preparando datos ruidosos {args.tipo} {args.nivel} (train/val on-demand)...", flush=True)
    train_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "train"), batch_size=args.batch, shuffle=True)
    val_noisy_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "val"), batch_size=args.batch)
    val_clean_ld = DataLoader(clean.val, batch_size=args.batch)
    test_clean_ld = DataLoader(clean.test, batch_size=args.batch)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                      weight_decay=args.weight_decay, scheduler="cosine",
                      scheduler_params={"t_max": args.epochs})
    ckpt = Path(args.ckpt_dir) / f"ckpt_{args.tipo}_{args.nivel}.pt"
    entry_id = f"gaussiano_live_{args.tipo}_{args.nivel}"
    datos = f"{args.tipo} {args.nivel}"
    trainer = Trainer(model, cfg, callbacks=[
        ModelCheckpoint(ckpt, every=1),
        LiveCurve(val_clean_ld, args.out, args.tipo, args.nivel),
        TrainingLogger(entry_id, args.epochs, modelo=f"CNN{channels} d{args.dropout}",
                       datos=datos, checkpoint=str(ckpt)),
    ])
    if args.resume:
        info = trainer.resume_from(ckpt)
        print(f"REANUDANDO desde época {info['epochs_done']} hasta {args.epochs} "
              f"(cosine t_max={args.epochs})...", flush=True)

    log_training(id=entry_id, estado="en_curso", modelo=f"CNN{channels} d{args.dropout}",
                 datos=datos, épocas=f"{trainer.start_epoch}/{args.epochs}", checkpoint=str(ckpt))

    print(f"Entrenando hasta {args.epochs} épocas (val ruidoso monitoriza la tarea)...", flush=True)
    trainer.fit(train_ld, val_noisy_ld)

    test_clean = trainer.evaluate(test_clean_ld)[0]
    val_best = max(trainer.history["val_accuracy"])
    log_training(id=entry_id, estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                 val=val_best, test=test_clean)
    print(f"\nHECHO | mejor val_ruidoso={val_best:.4f} | TEST_limpio={test_clean:.4f} | "
          f"curva en {args.out}", flush=True)


if __name__ == "__main__":
    main()
