"""Evalúa checkpoints ya entrenados sobre el TEST INVERTIDO (negativo) vs. el TEST limpio.

Rápido: una sola pasada por 10k imágenes por modelo. Mide cuánto cae la accuracy cuando se
invierte la polaridad fondo/trazo (hipótesis: las CNN de MNIST dependen de "trazo claro sobre
fondo oscuro" y colapsan al invertir).

Uso:
  python scripts/eval_inverted.py                       # set curado por defecto
  python scripts/eval_inverted.py --ckpts a=exp/a.pt b=exp/b.pt
"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnist.data import inverted_dataset, load_mnist
from nnist.models import build_model
from nnist.training import TrainConfig, Trainer

OUT = Path("reports/eval_inverted.png")


def default_ckpts() -> list[tuple[str, str]]:
    core = [
        ("limpio_20ep", "experiments/_noise_compare/ckpt_clean.pt"),
        ("gaussiano_20ep", "experiments/_noise_compare/ckpt_noisy.pt"),
        ("gaussiano_live_20ep", "experiments/_gaussiano_live/ckpt_gaussiano_nivel_3.pt"),
        ("k3_limpio", "experiments/_kernel_sweep/ckpt_k3.pt"),
        ("k5_limpio", "experiments/_kernel_sweep/ckpt_k5.pt"),
        ("k7_limpio", "experiments/_kernel_sweep/ckpt_k7.pt"),
    ]
    nt = [(f"nt_{Path(p).stem.replace('ckpt_', '')}", p)
          for p in sorted(glob("experiments/_noisetype_sweep/ckpt_*.pt"))]
    return [(l, p) for l, p in core + nt if Path(p).exists()]


def build_from_checkpoint(path, input_shape, num_classes):
    sd = torch.load(path, map_location="cpu")["model_state"]
    channels = [v.shape[0] for k, v in sd.items()
                if k.startswith("features") and k.endswith(".weight") and v.dim() == 4]
    kernel = int(sd["features.0.weight"].shape[2])
    batchnorm = any("running_mean" in k for k in sd)
    fc_hidden = sd["classifier.1.weight"].shape[0]
    model = build_model("cnn", input_shape=input_shape, num_classes=num_classes,
                        channels=channels, kernel_size=kernel, fc_hidden=fc_hidden,
                        batchnorm=batchnorm, dropout=0.2)
    model.load_state_dict(sd)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", default=None, help="pares etiqueta=ruta (opcional)")
    args = ap.parse_args()
    pairs = ([c.split("=", 1) for c in args.ckpts] if args.ckpts else default_ckpts())

    bundle = load_mnist()
    input_shape, num_classes = bundle.input_shape, bundle.num_classes
    xs, ys = zip(*[(x, int(y)) for x, y in bundle.test])
    clean_ds = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))
    inv_ds = inverted_dataset("test")

    def acc(model, ds):
        tr = Trainer(model, TrainConfig(device="cpu", batch_size=256))
        return tr.evaluate(DataLoader(ds, batch_size=256, shuffle=False))[0]

    rows = []
    print(f"{'modelo':<24}{'test_limpio':>12}{'test_invertido':>16}{'caida':>10}")
    print("-" * 62)
    for lab, path in pairs:
        model = build_from_checkpoint(path, input_shape, num_classes)
        a_clean = acc(model, clean_ds)
        a_inv = acc(model, inv_ds)
        rows.append((lab, a_clean, a_inv))
        print(f"{lab:<24}{a_clean:>12.4f}{a_inv:>16.4f}{a_clean - a_inv:>10.3f}", flush=True)

    # gráfica: accuracy invertido por modelo (con test limpio de referencia)
    rows_sorted = sorted(rows, key=lambda r: r[2])
    labs = [r[0] for r in rows_sorted]
    inv = [r[2] for r in rows_sorted]
    cln = [r[1] for r in rows_sorted]
    y = range(len(labs))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labs) + 2))
    ax.barh(list(y), inv, color="tab:purple", alpha=0.7, label="test invertido")
    ax.scatter(cln, list(y), color="black", s=18, zorder=3, label="test limpio (ref.)")
    ax.axvline(0.10, ls="--", color="red", lw=1, label="azar (0.10)")
    ax.set_yticks(list(y), labs, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("accuracy")
    ax.set_title("Modelos entrenados evaluados en dígitos INVERTIDOS")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"\nOK: {OUT}")


if __name__ == "__main__":
    main()
