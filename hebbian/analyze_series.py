"""Analiza una serie de snapshots (ep10/20/30/40/50) para estudiar la dinamica del entrenamiento.

Para cada snapshot (con un umbral de disparo theta comun) calcula, sobre todo el dataset:
  - cantidad de disparos por entrada: media, mediana, p10-p90, y % de entradas con MUY POCOS disparos
    (< --min-fire), que es lo que NO queremos.
  - active_pool: nº de neuronas que disparan para al menos una entrada.
  - si el dataset es pos+neg (mitades), separa la media de disparos de positivas vs negativas.

ESTABILIDAD (¿las neuronas de una entrada se mantienen al entrenar mas?):
  - entre snapshots CONSECUTIVOS: solape Jaccard del conjunto de neuronas que disparan por entrada, y
    winner_match = fraccion de entradas cuya neurona ganadora (argmax) es la MISMA.
  - de cada snapshot vs el FINAL (ep50): mismo par de metricas -> mide si converge hacia el estado final.

Uso:
    python hebbian/analyze_series.py --dir experiments/series_posneg_r1_g15 \
        --dataset data/processed/lines_hebbian/lines_posneg.npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ


def load_X(path: Path) -> np.ndarray:
    imgs = np.load(path)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    return X


def jaccard(Fa: np.ndarray, Fb: np.ndarray) -> float:
    inter = (Fa & Fb).sum(1)
    union = (Fa | Fb).sum(1)
    return float(np.divide(inter, union, out=np.zeros(len(inter)), where=union > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Analiza estabilidad y disparos de una serie de snapshots")
    ap.add_argument("--dir", type=Path, default=Path("experiments/series_posneg_r1_g15"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/lines_hebbian/lines_posneg.npz"))
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--min-fire", type=int, default=5, help="por debajo de esto, 'muy pocos disparos'")
    ap.add_argument("--posneg", action=argparse.BooleanOptionalAction, default=False,
                    help="dataset = mitad pos + mitad neg (separa medias)")
    args = ap.parse_args()

    paths = sorted(args.dir.glob("model_ep*.npz"))
    if not paths:
        raise SystemExit(f"no hay snapshots en {args.dir}")
    layers = [CompetitiveLayer.load(p) for p in paths]
    eps = [L.epochs_trained for L in layers]
    X = load_X(args.dataset)
    Xn = layers[0]._normalize_rows(X)
    thr = args.threshold
    half = len(X) // 2

    print(f"snapshots: {[p.name for p in paths]}  (epocas {eps})")
    print(f"dataset {args.dataset} ({len(X)} entradas)  theta={thr}\n")

    Fs, Ws, per = [], [], []
    for L, ep in zip(layers, eps):
        A = Xn @ L.W.T
        F = A >= thr
        fired = F.sum(1)
        Fs.append(F)
        Ws.append(A.argmax(1))
        rec = {
            "epoch": ep,
            "mean_fired": round(float(fired.mean()), 1),
            "median_fired": int(np.median(fired)),
            "p10": int(np.percentile(fired, 10)),
            "p90": int(np.percentile(fired, 90)),
            "pct_too_few": round(100 * float((fired < args.min_fire).mean()), 1),
            "active_pool": int(F.any(0).sum()),
        }
        if args.posneg:
            rec["fired_pos"] = round(float(fired[:half].mean()), 1)
            rec["fired_neg"] = round(float(fired[half:].mean()), 1)
        per.append(rec)

    print("== POR SNAPSHOT ==")
    hdr = ["epoch", "mean_fired", "median_fired", "p10", "p90", "pct_too_few", "active_pool"]
    if args.posneg:
        hdr += ["fired_pos", "fired_neg"]
    print("  ".join(f"{h:>12}" for h in hdr))
    for r in per:
        print("  ".join(f"{r[h]:>12}" for h in hdr))

    print("\n== ESTABILIDAD (consecutivos) ==")
    print(f"{'par':>14} | {'jaccard_disp':>12} | {'winner_match':>12}")
    stab = []
    for i in range(1, len(layers)):
        j = jaccard(Fs[i - 1], Fs[i])
        wm = float((Ws[i - 1] == Ws[i]).mean())
        stab.append({"from": eps[i - 1], "to": eps[i], "jaccard": round(j, 3), "winner_match": round(wm, 3)})
        print(f"{f'{eps[i-1]}->{eps[i]}':>14} | {j:>12.3f} | {wm:>12.3f}")

    print("\n== CONVERGENCIA (cada snapshot vs FINAL ep{}) ==".format(eps[-1]))
    print(f"{'epoca':>14} | {'jaccard_disp':>12} | {'winner_match':>12}")
    conv = []
    for i in range(len(layers)):
        j = jaccard(Fs[i], Fs[-1])
        wm = float((Ws[i] == Ws[-1]).mean())
        conv.append({"epoch": eps[i], "jaccard_final": round(j, 3), "winner_match_final": round(wm, 3)})
        print(f"{eps[i]:>14} | {j:>12.3f} | {wm:>12.3f}")

    # csv
    with open(args.dir / "analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        ax[0].plot(eps, [r["mean_fired"] for r in per], "-o", label="media")
        ax[0].plot(eps, [r["median_fired"] for r in per], "-o", label="mediana")
        if args.posneg:
            ax[0].plot(eps, [r["fired_pos"] for r in per], "--s", label="pos")
            ax[0].plot(eps, [r["fired_neg"] for r in per], "--s", label="neg")
        ax[0].set(title="Disparos por entrada vs epoca", xlabel="epoca", ylabel="neuronas disparadas")
        ax[0].legend(fontsize=8)
        ax[1].plot([s["to"] for s in stab], [s["jaccard"] for s in stab], "-o", label="jaccard disparo")
        ax[1].plot([s["to"] for s in stab], [s["winner_match"] for s in stab], "-o", label="winner match")
        ax[1].set(title="Estabilidad consecutiva (1=estable)", xlabel="epoca", ylabel="fraccion", ylim=(0, 1))
        ax[1].legend(fontsize=8)
        ax[2].plot(eps, [c["jaccard_final"] for c in conv], "-o", label="jaccard vs final")
        ax[2].plot(eps, [c["winner_match_final"] for c in conv], "-o", label="winner match vs final")
        ax[2].plot(eps, [r["pct_too_few"] / 100 for r in per], "--s", label="frac pocos disparos")
        ax[2].set(title="Convergencia al estado final", xlabel="epoca", ylabel="fraccion", ylim=(0, 1))
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.dir / "analysis.png", dpi=110)
        plt.close(fig)
        print(f"\ngrafica -> {args.dir / 'analysis.png'}")
    except Exception as e:
        print(f"(sin grafica: {e})")
    print(f"csv     -> {args.dir / 'analysis.csv'}")


if __name__ == "__main__":
    main()
