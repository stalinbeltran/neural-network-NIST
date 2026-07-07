"""Control para AISLAR el invertido: entrena CNN con mezcla normal + gaussiano n3 (SIN invertido).

Comparte receta con el bipolar+gaussiano (CNN[16,32,64] d0.2, cosine, lr 0.001, wd 1e-4) para que
la ÚNICA diferencia sea la ausencia del stream invertido. train = normal (48k) + gaussiano n3 (48k)
= 96k mezclados por batch. Monitoriza val en limpio y en gaussiano. Reanudable, con avance por época.

  python scripts/train_normal_gaussiano.py --epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from nnist.data import load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer, TrainingLogger
from nnist.utils import log_training, set_seed


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return xs, ys


class PrintEpoch(Callback):
    """Imprime val limpio + val gaussiano al terminar cada época (avance progresivo)."""

    def __init__(self, val_gauss_loader):
        self.val_gauss = val_gauss_loader

    def on_epoch_end(self, trainer, epoch, metrics):
        h = trainer.history
        vg = h.setdefault("val_gaussiano", [])
        vg.append(trainer.evaluate(self.val_gauss)[0])
        print(f"  época {epoch + 1}: loss={h['train_loss'][-1]:.4f} | "
              f"val_limpio={metrics['val_accuracy']:.4f} | val_gauss={vg[-1]:.4f}", flush=True)


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
    ap.add_argument("--ckpt-dir", default="experiments/_normal_gaussiano")
    ap.add_argument("--resume", action="store_true")
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
    print(f"Datos: normal + {args.tipo} {args.nivel} (48k*2=96k, SIN invertido)...", flush=True)
    train_mix = ConcatDataset([clean.train, noisy_dataset(args.tipo, args.nivel, "train")])
    train_ld = DataLoader(train_mix, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    val_norm_ld = DataLoader(clean.val, batch_size=args.batch)
    val_gauss_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "val"), batch_size=args.batch)
    test_norm_ld = DataLoader(clean.test, batch_size=args.batch)
    test_gauss_ld = DataLoader(noisy_dataset(args.tipo, args.nivel, "test"), batch_size=args.batch)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                      weight_decay=args.weight_decay, scheduler="cosine",
                      scheduler_params={"t_max": args.epochs})
    ckpt = Path(args.ckpt_dir) / "ckpt_normal_gaussiano.pt"
    entry_id = "normal_gaussiano"
    datos = f"normal+{args.tipo}_{args.nivel.split('_')[1]}"
    trainer = Trainer(model, cfg, callbacks=[
        ModelCheckpoint(ckpt, every=1),
        PrintEpoch(val_gauss_ld),
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
    tg = trainer.evaluate(test_gauss_ld)[0]
    log_training(id=entry_id, estado="hecho", épocas=f"{args.epochs}/{args.epochs}",
                 val=max(trainer.history["val_accuracy"]), test=f"n {tn:.3f}/g {tg:.3f}")
    print(f"\nHECHO | TEST normal={tn:.4f} | gaussiano_n3={tg:.4f} | checkpoint={ckpt}", flush=True)


if __name__ == "__main__":
    main()
