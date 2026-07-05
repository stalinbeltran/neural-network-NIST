"""Compara 2 CNN idénticas: una entrenada con datos LIMPIOS, otra con datos RUIDOSOS
(gaussiano sigma=0.5 = nivel_3). VAL y TEST son SIEMPRE LIMPIOS (solo el train lleva ruido).

Usa el Trainer con `ModelCheckpoint` -> es REANUDABLE. Cada modelo guarda su checkpoint en
experiments/_noise_compare/{ckpt_clean,ckpt_noisy}.pt.

  # entrenar 1 época (deja checkpoints)
  python scripts/noise_train_compare.py --epochs 1 --checkpoint-every 1
  # reanudar hasta 2 épocas
  python scripts/noise_train_compare.py --epochs 2 --checkpoint-every 1 --resume
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nnist.data import load_mnist, noisy_dataset
from nnist.models import build_model
from nnist.training import Callback, ModelCheckpoint, TrainConfig, Trainer
from nnist.utils import set_seed

CKPT_DIR = Path("experiments") / "_noise_compare"


class PrintEpoch(Callback):
    """Imprime val (limpio) al terminar cada época, para ver el avance."""
    def __init__(self, tag: str):
        self.tag = tag

    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"  [{self.tag:<8}] época {epoch + 1}: val_limpio={metrics['val_accuracy']:.4f}", flush=True)


def _make_trainer(model, cfg, ckpt_path, every, tag) -> Trainer:
    return Trainer(model, cfg, callbacks=[ModelCheckpoint(ckpt_path, every=every), PrintEpoch(tag)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tipo", default="gaussiano")
    ap.add_argument("--nivel", default="nivel_3")
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="reanudar desde los checkpoints existentes")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    channels = [16, 32, 64]            # la CNN con menos pesos (98.442 params)
    input_shape = (1, 28, 28)
    clean_ckpt = CKPT_DIR / "ckpt_clean.pt"
    noisy_ckpt = CKPT_DIR / "ckpt_noisy.pt"

    print("Preparando datos...", flush=True)
    clean = load_mnist()
    noisy_tr = noisy_dataset(args.tipo, args.nivel, "train")
    clean_tr = DataLoader(clean.train, batch_size=args.batch, shuffle=True)
    noisy_ld = DataLoader(noisy_tr, batch_size=args.batch, shuffle=True)
    val_ld = DataLoader(clean.val, batch_size=args.batch)       # VAL limpio
    test_ld = DataLoader(clean.test, batch_size=args.batch)     # TEST limpio

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch)
    set_seed(args.seed); m_clean = build_model("cnn", input_shape=input_shape, num_classes=10, channels=channels)
    set_seed(args.seed); m_noisy = build_model("cnn", input_shape=input_shape, num_classes=10, channels=channels)

    t_clean = _make_trainer(m_clean, cfg, clean_ckpt, args.checkpoint_every, "LIMPIO")
    t_noisy = _make_trainer(m_noisy, cfg, noisy_ckpt, args.checkpoint_every, "RUIDOSO")

    if args.resume:
        c = t_clean.resume_from(clean_ckpt)
        n = t_noisy.resume_from(noisy_ckpt)
        print(f"REANUDANDO desde época limpio={c['epochs_done']} / ruidoso={n['epochs_done']} "
              f"hasta {args.epochs}", flush=True)

    print(f"Modelo: CNN {channels} | params={m_clean.count_params()['params_total']:,} | "
          f"ruido train: {args.tipo} {args.nivel} | eval: LIMPIO", flush=True)

    print("Entrenando modelo LIMPIO...", flush=True)
    t_clean.fit(clean_tr, val_ld)
    print("Entrenando modelo RUIDOSO...", flush=True)
    t_noisy.fit(noisy_ld, val_ld)

    hc, hn = t_clean.history["val_accuracy"], t_noisy.history["val_accuracy"]
    print(f"\n{'época':<7}{'val_limpio':<12}{'val_ruidoso':<13}{'Δ (pp)':<8}", flush=True)
    for i in range(min(len(hc), len(hn))):
        print(f"{i + 1:<7}{hc[i]:<12.4f}{hn[i]:<13.4f}{(hc[i] - hn[i]) * 100:<8.2f}", flush=True)

    tc = t_clean.evaluate(test_ld)[0]
    tn = t_noisy.evaluate(test_ld)[0]
    print(f"\nTEST (limpio) -> entrenado_LIMPIO={tc:.4f}  entrenado_RUIDOSO={tn:.4f}  "
          f"Δ={(tc - tn) * 100:.2f} pp", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / "history.json").write_text(json.dumps(
        {"val_clean": hc, "val_noisy": hn, "test_clean": tc, "test_noisy": tn,
         "epochs_done": t_clean._epochs_done, "channels": channels,
         "tipo": args.tipo, "nivel": args.nivel}, indent=2), encoding="utf-8")

    plot_history(hc, hn, args.tipo, args.nivel)


def plot_history(hc, hn, tipo, nivel, out="reports/noise_train_compare.png") -> None:
    """Gráfica de val (limpio) por época: modelo entrenado en limpio vs en ruidoso."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(sin gráfica: {e})", flush=True)
        return
    ep = range(1, len(hc) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, hc, "o-", label="entrenado LIMPIO")
    ax.plot(ep, hn, "o-", label=f"entrenado RUIDOSO ({tipo} {nivel})")
    ax.set_xlabel("época"); ax.set_ylabel("val_accuracy (evaluado en LIMPIO)")
    ax.set_title("Entrenamiento limpio vs ruidoso (test/val limpios)")
    ax.grid(True, alpha=0.3); ax.legend()
    if len(hc) <= 12:
        ax.set_xticks(list(ep))
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Gráfica: {out}", flush=True)


if __name__ == "__main__":
    main()
