"""Entrena la capa competitiva sobre el dataset de rectas y GRABA el progreso por iteracion.

Cada "iteracion" = una pasada completa (epoca) sobre las ~1000 rectas en orden aleatorio.
Despues de CADA iteracion se guarda:
  - frames/rf_epoch_XXX.png : mosaico 50x50 con el campo receptivo (los pesos) de cada neurona,
    es decir "que patron ha aprendido a disparar cada una". Ver como se forman las rectas = ver
    el aprendizaje.
  - frames/act_epoch_XXX.png : ante unas pocas rectas de muestra, el mapa 50x50 de QUE NEURONAS
    se disparan (activacion). Muestra el disparo evolucionando.
  - weights/epoch_XXX.npz   : snapshot de los pesos (cada `--snapshot-every` epocas y al final).
  - metrics.csv             : una fila por epoca (activacion media del ganador, cobertura, etc.).
Al terminar se montan dos GIF (campos receptivos y activaciones) y una grafica de metricas.

Todo se escribe en experiments/hebbian_lines_<timestamp>/ (registro completo del progreso).

Uso:
    python hebbian/train.py                       # 50x50 salidas, 40 epocas, regla 'above_mean'
    python hebbian/train.py --epochs 60 --n-out 2500 --rule above_mean
    python hebbian/train.py --rule wta            # winner-take-all (competicion dura)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ, SIZE, generate

GRID = 50                      # salida 50x50 = 2500 neuronas
GAP = 1                        # separacion en px entre tiles del mosaico
N_SAMPLES = 6                  # rectas de muestra para el mapa de activacion


# ----------------------------------------------------------------------- datos
def load_lines(path: Path, n: int, seed: int) -> np.ndarray:
    """Carga (o genera, solo para el dataset de rectas por defecto) un set de entradas.

    Devuelve (N, n_in) float32 en [0,1]. `path` puede apuntar a CUALQUIER .npz con una clave
    `images` de forma (N, H, W) o (N, D) -> asi se reanuda el entrenamiento con otros sets."""
    if not path.exists():
        if path == LINES_NPZ:
            print("dataset de rectas no encontrado; generando...")
            blob = generate(n, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, images=blob["images"], angles=blob["angles"])
            imgs = blob["images"]
        else:
            raise FileNotFoundError(f"no existe el dataset {path}")
    else:
        imgs = np.load(path)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:                          # normaliza uint8 [0,255] -> [0,1]
        X /= 255.0
    return X


# ----------------------------------------------------------------------- visualizacion
def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def receptive_field_grid(layer: CompetitiveLayer) -> np.ndarray:
    """Mosaico uint8 (GRID*28+gaps) con los pesos de cada neurona, normalizados por tile."""
    rf = layer.receptive_fields(SIZE, SIZE)                 # (2500, 28, 28)
    tile = SIZE + GAP
    canvas = np.full((GRID * tile + GAP, GRID * tile + GAP), 40, dtype=np.uint8)
    for i in range(layer.n_out):
        r, c = divmod(i, GRID)
        img = (_norm01(rf[i]) * 255).astype(np.uint8)
        y, x = GAP + r * tile, GAP + c * tile
        canvas[y:y + SIZE, x:x + SIZE] = img
    return canvas


def winners_grid(layer: CompetitiveLayer, winners: np.ndarray) -> np.ndarray:
    """Mapa 50x50 mostrando SOLO las ganadoras (su campo receptivo); las perdedoras en negro."""
    rf = layer.receptive_fields(SIZE, SIZE)
    tile = SIZE + GAP
    canvas = np.zeros((GRID * tile + GAP, GRID * tile + GAP), dtype=np.uint8)
    for i in winners:
        r, c = divmod(int(i), GRID)
        y, x = GAP + r * tile, GAP + c * tile
        canvas[y:y + SIZE, x:x + SIZE] = (_norm01(rf[int(i)]) * 255).astype(np.uint8)
    return canvas


def current_winners(layer: CompetitiveLayer, Xn: np.ndarray) -> np.ndarray:
    """Indices de neuronas que ganan (argmax) al menos una entrada, con los pesos actuales."""
    return np.unique(layer.activate_batch(Xn).argmax(axis=1))


def _annotate(canvas: np.ndarray, text: str) -> Image.Image:
    """Anade una banda superior con el texto (epoca + metricas)."""
    img = Image.fromarray(canvas).convert("L")
    band = 20
    out = Image.new("L", (img.width, img.height + band), 0)
    out.paste(img, (0, band))
    ImageDraw.Draw(out).text((4, 5), text, fill=255)
    return out


def activation_panel(layer: CompetitiveLayer, samples: np.ndarray) -> Image.Image:
    """Para cada recta de muestra: [entrada 28x28 | mapa de activacion 50x50], apiladas."""
    up = 3                                                   # factor de escala para ver mejor
    rows = []
    Xs = layer._normalize_rows(samples)
    A = layer.activate_batch(Xs)                            # (N_SAMPLES, 2500)
    for k in range(len(samples)):
        inp = (_norm01(samples[k].reshape(SIZE, SIZE)) * 255).astype(np.uint8)
        inp = np.asarray(Image.fromarray(inp).resize((SIZE * up * 2, SIZE * up * 2), Image.NEAREST))
        act = (_norm01(A[k].reshape(GRID, GRID)) * 255).astype(np.uint8)
        act = np.asarray(Image.fromarray(act).resize((GRID * up, GRID * up), Image.NEAREST))
        h = max(inp.shape[0], act.shape[0])
        row = np.full((h, inp.shape[1] + 8 + act.shape[1]), 30, dtype=np.uint8)
        row[:inp.shape[0], :inp.shape[1]] = inp
        row[:act.shape[0], inp.shape[1] + 8:] = act
        rows.append(row)
    gap = np.full((6, rows[0].shape[1]), 0, dtype=np.uint8)
    stacked = rows[0]
    for r in rows[1:]:
        stacked = np.vstack([stacked, gap, r])
    return Image.fromarray(stacked)


def save_gif(frames: list[Image.Image], path: Path, max_side: int = 760, ms: int = 250) -> None:
    if not frames:
        return
    scaled = []
    for f in frames:
        if max(f.size) > max_side:
            s = max_side / max(f.size)
            f = f.resize((int(f.width * s), int(f.height * s)), Image.NEAREST)
        scaled.append(f.convert("P"))
    scaled[0].save(path, save_all=True, append_images=scaled[1:], duration=ms, loop=0)


def plot_metrics(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r["mean_winner_activation"] for r in rows], "-o", ms=3)
    ax[0].set(title="Activacion media del ganador", xlabel="iteracion (epoca)", ylabel="cos sim")
    ax[1].plot(ep, [r["coverage"] for r in rows], "-o", ms=3, label="cobertura")
    ax[1].plot(ep, [r["dead_units"] / (GRID * GRID) for r in rows], "-o", ms=3, label="neuronas muertas")
    ax[1].set(title="Uso de las neuronas", xlabel="iteracion (epoca)", ylabel="fraccion")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------- entrenamiento
def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena la capa competitiva sobre rectas y graba el progreso")
    ap.add_argument("--epochs", type=int, default=40, help="numero de iteraciones (pasadas)")
    ap.add_argument("--n-out", type=int, default=GRID * GRID, help="neuronas de salida (default 2500 = 50x50)")
    ap.add_argument("--rule", choices=["above_mean", "softmax", "wta"], default="above_mean")
    ap.add_argument("--temperature", type=float, default=0.1, help="solo para rule=softmax")
    ap.add_argument("--anti", type=float, default=1.0,
                    help="fuerza anti-Hebbiana: cuanto debilitan las neuronas lejanas (0 = desactivado)")
    ap.add_argument("--lr0", type=float, default=0.3, help="learning rate inicial")
    ap.add_argument("--lr-min", type=float, default=0.02, help="learning rate final")
    ap.add_argument("--n", type=int, default=1000, help="imagenes si hay que generar el dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=5, help="cada cuantas epocas guardar pesos .npz")
    ap.add_argument("--frame-every", type=int, default=25,
                    help="captura un fotograma de la animacion densa cada N muestras (0 = solo por epoca)")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="tope de fotogramas de la animacion densa (evita GIF gigante)")
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ,
                    help="npz con clave 'images'; permite reanudar con OTRO set de entradas")
    ap.add_argument("--resume", type=Path, default=None,
                    help="model.npz de una corrida previa: continua su entrenamiento en vez de empezar de cero")
    # --- neuronas inhibidoras (reduccion de pesos; el algoritmo base solo incrementa) ---
    ap.add_argument("--inhib", action="store_true", help="activa las neuronas inhibidoras")
    ap.add_argument("--inhib-spacing", type=int, default=5, help="separacion de los inhibidores en el mapa")
    ap.add_argument("--inhib-radius", type=int, default=8, help="alcance (radio) de cada inhibidor")
    ap.add_argument("--inhib-metric", choices=["cheby", "euclid", "manhattan"], default="cheby")
    ap.add_argument("--fire-threshold", type=float, default=0.40, help="umbral de disparo (para contar disparos)")
    ap.add_argument("--inhib-K", type=float, default=0.10, help="bajo este valor el inhibidor no reduce nada")
    ap.add_argument("--inhib-gain", type=float, default=1.0, help="cuanto reduce por unidad de exceso")
    ap.add_argument("--inhib-mode", choices=["fraction", "hinge", "sigmoid"], default="fraction")
    ap.add_argument("--reinforce-gain", type=float, default=1.0,
                    help="ganancia de activacion: multiplica el refuerzo (incremento de pesos)")
    args = ap.parse_args()

    if args.n_out != GRID * GRID:
        print(f"aviso: la visualizacion asume {GRID}x{GRID}={GRID*GRID} neuronas; con {args.n_out} el mosaico no cuadrara")

    X = load_lines(args.dataset, args.n, args.seed)
    print(f"dataset: {X.shape[0]} entradas de {X.shape[1]} pixeles ({args.dataset})")

    if args.resume:
        layer = CompetitiveLayer.load(args.resume)
        if layer.n_in != X.shape[1]:
            raise ValueError(f"la red guardada espera {layer.n_in} entradas pero el dataset tiene {X.shape[1]}")
        print(f"reanudando desde {args.resume}: {layer.epochs_trained} iteraciones ya entrenadas, "
              f"rule={layer.rule}")
    else:
        layer = CompetitiveLayer(X.shape[1], args.n_out, rule=args.rule,
                                 temperature=args.temperature, anti=args.anti,
                                 reinforce_gain=args.reinforce_gain, seed=args.seed)

    if args.inhib:                                       # activa/reconfigura la inhibicion (fresco o resume)
        n_inh = layer.configure_inhibition(spacing=args.inhib_spacing, radius=args.inhib_radius,
                                           metric=args.inhib_metric, fire_threshold=args.fire_threshold,
                                           K=args.inhib_K, gain=args.inhib_gain, mode=args.inhib_mode)
        print(f"inhibicion ON: {n_inh} inhibidores (cada {args.inhib_spacing}, radio {args.inhib_radius} "
              f"{args.inhib_metric})  theta={args.fire_threshold} K={args.inhib_K} gain={args.inhib_gain} "
              f"modo={args.inhib_mode}")
    elif layer.inhib_on:
        print(f"inhibicion ON (heredada del modelo): {len(layer._inhib_regions)} inhibidores")
    rng = np.random.default_rng(args.seed + 1)

    run_dir = Path("experiments") / f"hebbian_lines_{time.strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    print(f"grabando progreso en {run_dir}")

    # rectas de muestra FIJAS para el panel de activacion (mismas en todas las epocas)
    sample_idx = np.linspace(0, len(X) - 1, N_SAMPLES).astype(int)
    samples = X[sample_idx]

    rf_frames: list[Image.Image] = []
    act_frames: list[Image.Image] = []
    dense_frames: list[Image.Image] = []
    winner_frames: list[Image.Image] = []
    metrics: list[dict] = []
    DENSE_SIDE = 480

    def shrink(img: Image.Image, side: int = DENSE_SIDE) -> Image.Image:
        if max(img.size) > side:
            s = side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.NEAREST)
        return img

    Xn = layer._normalize_rows(X)          # entradas normalizadas (fijo; solo cambia W)

    def winner_frame() -> Image.Image:
        w = current_winners(layer, Xn)
        text = f"muestras {samples_seen[0]}  ganadoras {len(w)}"
        return shrink(_annotate(winners_grid(layer, w), text))

    # animaciones: contador de muestras acumuladas (continua si se reanuda)
    samples_seen = [layer.epochs_trained * len(X)]
    dense_frames.append(shrink(_annotate(receptive_field_grid(layer),
                                          f"muestras {samples_seen[0]}  (pesos aleatorios iniciales)")))
    winner_frames.append(winner_frame())

    # frame inicial (epoca 0 = pesos aleatorios, antes de aprender)
    def snapshot(ep: int, m: dict | None) -> None:
        info = f"iter {ep:03d} (total {layer.epochs_trained})"
        if m:
            info += f"  act={m['mean_winner_activation']:.3f}  cobertura={m['coverage']:.2f}  muertas={m['dead_units']}"
        rf = _annotate(receptive_field_grid(layer), info)
        rf.save(run_dir / "frames" / f"rf_epoch_{ep:03d}.png")
        rf_frames.append(rf)
        act = activation_panel(layer, samples)
        act.save(run_dir / "frames" / f"act_epoch_{ep:03d}.png")
        act_frames.append(act)

    snapshot(0, None)

    for ep in range(1, args.epochs + 1):
        frac = (ep - 1) / max(args.epochs - 1, 1)
        lr = args.lr0 * (args.lr_min / args.lr0) ** frac        # decae exponencial lr0 -> lr_min

        def on_frame(ep=ep, lr=lr) -> None:                     # captura DENTRO de la epoca
            if len(dense_frames) >= args.max_frames:
                return
            samples_seen[0] += args.frame_every
            img = _annotate(receptive_field_grid(layer),
                            f"muestras {samples_seen[0]}  iter {ep:03d}  lr={lr:.3f}")
            dense_frames.append(shrink(img))
            winner_frames.append(winner_frame())                # mapa 50x50 de solo ganadoras

        m = layer.learn_epoch(X, lr, rng,
                              on_frame=on_frame if args.frame_every else None,
                              frame_every=args.frame_every)
        m["epoch"] = ep
        m["lr"] = lr
        metrics.append(m)
        snapshot(ep, m)
        if ep % args.snapshot_every == 0 or ep == args.epochs:
            np.savez_compressed(run_dir / "weights" / f"epoch_{ep:03d}.npz", W=layer.W)
            layer.save(run_dir / "model.npz")           # estado completo reanudable (se sobrescribe)
        print(f"iter {ep:03d}/{args.epochs}  lr={lr:.3f}  act={m['mean_winner_activation']:.3f}  "
              f"cobertura={m['coverage']:.2f}  ganadores_unicos={m['unique_winners']}  "
              f"muertas={m['dead_units']}  disparadas/entrada={m['mean_fired']:.1f}")

    # registro de metricas
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "lr", "mean_winner_activation", "coverage",
                                          "unique_winners", "dead_units", "mean_fired"])
        w.writeheader()
        for m in metrics:
            w.writerow({k: m[k] for k in w.fieldnames})

    save_gif(winner_frames, run_dir / "winners_process.gif", max_side=DENSE_SIDE, ms=150)
    save_gif(dense_frames, run_dir / "learning_process.gif", max_side=DENSE_SIDE, ms=120)
    save_gif(rf_frames, run_dir / "progress_receptive_fields.gif")
    save_gif(act_frames, run_dir / "progress_activations.gif")
    plot_metrics(metrics, run_dir / "metrics.png")

    print("\nlisto. Revisa:")
    print(f"  {run_dir / 'winners_process.gif'}             <- ANIMACION: las GANADORAS a medida que entrenan")
    print(f"  {run_dir / 'learning_process.gif'}            <- ANIMACION DENSA: todo el proceso desde el inicio")
    print(f"  {run_dir / 'progress_receptive_fields.gif'}   <- como se forman las rectas aprendidas (por epoca)")
    print(f"  {run_dir / 'progress_activations.gif'}        <- que neuronas se disparan por muestra")
    print(f"  {run_dir / 'metrics.png'} y metrics.csv       <- progreso numerico")
    print(f"  {run_dir / 'frames'}                          <- PNG de cada iteracion")
    print(f"  {run_dir / 'model.npz'}                       <- red guardada (reanudable)")
    print(f"\nContinuar entrenando esta red (mismo u otro set):")
    print(f"  python hebbian/train.py --resume {run_dir / 'model.npz'} --epochs 20")
    print(f"  python hebbian/train.py --resume {run_dir / 'model.npz'} --dataset OTRO.npz --epochs 20")


if __name__ == "__main__":
    main()
