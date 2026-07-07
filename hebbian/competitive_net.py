"""Red de UNA capa con aprendizaje competitivo / Hebbiano (no supervisado).

Idea (segun lo pedido):
  1. Se crea la capa con pesos ALEATORIOS: W de forma (n_out, n_in).
  2. Se presenta una entrada x. Los pesos aleatorios hacen que ciertas neuronas se "disparen"
     (o esten cerca de dispararse) y otras queden lejos.
  3. Para cada neurona que esta MAS CERCA de dispararse, se mueven sus pesos hacia la entrada,
     de modo que la proxima vez sea un poco mas probable que se dispare ante entradas parecidas.

"Cerca de dispararse" = activacion alta. Trabajamos con activacion = SIMILITUD COSENO:
normalizamos cada entrada y cada fila de pesos a norma 1, asi a_i = w_i . x esta en [-1, 1] y es
comparable entre neuronas. Disparar ~ activacion cercana a 1.

Regla de aprendizaje (el algoritmo base SOLO INCREMENTA pesos):
  Para cada neurona se calcula un "gate" g_i >= 0 segun su cercania a disparar. Las neuronas cerca
  de dispararse (activacion por encima de la media) refuerzan sus pesos hacia la entrada presente:

    Delta w_ij = lr * g_i * x_j          (g_i >= 0)

  Las neuronas lejanas NO se tocan. Tras cada muestra se renormaliza w_i a norma 1 (mantiene la
  activacion en escala coseno y evita que los pesos exploten).

  Variantes (--rule):
    - 'above_mean' (def): g_i = tanh(relu(a - media(a)) / std). Encima de la media refuerza.
    - 'softmax'         : g_i = relu(tanh((a - media(a)) / T)).
    - 'wta'             : solo la mas activa refuerza.

REDUCCION de pesos = neuronas INHIBIDORAS (unica via):
  El algoritmo base nunca reduce pesos. Eso lo hacen neuronas inhibidoras SUPERPUESTAS, colocadas en
  una rejilla regular del mapa (cada `inhib_spacing`), cada una con una REGION de alcance `inhib_radius`
  (metrica `cheby`=cuadrado por defecto). Cada inhibidor cuenta cuantas de sus neuronas disparan
  (activacion >= `fire_threshold`); si la FRACCION supera `inhib_K`, debilita las conexiones
  entrada-activa -> neurona-disparada en proporcion al exceso (`inhib_gain`), regulando el exceso de
  disparos. Por debajo de `inhib_K` no reduce nada. Es opt-in (`inhib_on`). Ver configure_inhibition().

Implementado en numpy: rapido y sin dependencias de framework.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class CompetitiveLayer:
    def __init__(self, n_in: int, n_out: int, *, rule: str = "above_mean",
                 temperature: float = 0.1, anti: float = 1.0, reinforce_gain: float = 1.0,
                 grid_h: int | None = None, grid_w: int | None = None,
                 inhib_on: bool = False, inhib_spacing: int = 5, inhib_offset: int | None = None,
                 inhib_radius: int = 8, inhib_metric: str = "cheby", fire_threshold: float = 0.40,
                 inhib_K: float = 0.10, inhib_gain: float = 1.0, inhib_mode: str = "fraction",
                 seed: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.rule = rule
        self.temperature = temperature
        self.anti = anti                       # (legado, ya no se usa: el algoritmo base solo refuerza)
        self.reinforce_gain = reinforce_gain   # GANANCIA DE ACTIVACION: multiplica el refuerzo
        # geometria del mapa (para UBICAR las neuronas inhibidoras). Por defecto cuadrado sqrt(n_out).
        if grid_h is None or grid_w is None:
            s = int(round(np.sqrt(n_out)))
            grid_h = grid_w = s
        if grid_h * grid_w != n_out:
            raise ValueError(f"grid {grid_h}x{grid_w} no cuadra con n_out={n_out}")
        self.grid_h, self.grid_w = grid_h, grid_w
        self._nr = np.arange(n_out) // grid_w    # fila de cada neurona en el mapa
        self._nc = np.arange(n_out) % grid_w     # columna de cada neurona en el mapa
        rng = np.random.default_rng(seed)
        # pesos aleatorios pequenos, luego normalizados por fila a norma 1
        W = rng.standard_normal((n_out, n_in)).astype(np.float32)
        self.W = self._normalize_rows(W)
        # contador de "victorias" por neurona (para diagnostico: cobertura, neuronas muertas)
        self.win_count = np.zeros(n_out, dtype=np.int64)
        self.epochs_trained = 0                # iteraciones acumuladas (persiste entre reanudaciones)
        # --- inhibicion lateral local (neuronas inhibidoras SUPERPUESTAS) ---
        # Se guardan los parametros siempre (aunque este apagada) para persistir/reanudar.
        self.inhib_spacing = inhib_spacing
        self.inhib_offset = inhib_spacing // 2 if inhib_offset is None else inhib_offset
        self.inhib_radius = inhib_radius
        self.inhib_metric = inhib_metric
        self.fire_threshold = fire_threshold     # umbral de disparo (para contar disparos)
        self.inhib_K = inhib_K                    # bajo este valor el inhibidor NO reduce nada
        self.inhib_gain = inhib_gain              # cuanto reduce por unidad de exceso
        self.inhib_mode = inhib_mode              # 'fraction' | 'hinge' | 'sigmoid'
        self.inhib_on = False
        self._inhib_regions: list[np.ndarray] = []
        if inhib_on:
            self.configure_inhibition(spacing=inhib_spacing, offset=self.inhib_offset,
                                      radius=inhib_radius, metric=inhib_metric,
                                      fire_threshold=fire_threshold, K=inhib_K, gain=inhib_gain,
                                      mode=inhib_mode)

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _normalize_rows(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(norms, eps)

    @staticmethod
    def _normalize_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return x / max(float(np.linalg.norm(x)), eps)

    # ------------------------------------------------------------------ forward
    def activate(self, x_unit: np.ndarray) -> np.ndarray:
        """Activaciones (similitud coseno) de todas las neuronas ante x ya normalizada."""
        return self.W @ x_unit

    def activate_batch(self, X_unit: np.ndarray) -> np.ndarray:
        """(N, n_in) normalizado -> (N, n_out) activaciones."""
        return X_unit @ self.W.T

    # ------------------------------------------------------------------ gate (solo refuerzo)
    def _gate(self, a: np.ndarray) -> np.ndarray:
        """Gate g_i >= 0 POR NEURONA. El algoritmo base SOLO INCREMENTA pesos: las neuronas cerca
        de disparar (activacion alta) refuerzan sus pesos hacia la entrada presente; las lejanas se
        quedan a 0 (no se tocan). TODA reduccion de pesos queda a cargo de las neuronas inhibidoras.
        La magnitud crece con lo por encima de la media que esta la activacion."""
        if self.rule == "wta":
            g = np.zeros_like(a)
            g[int(np.argmax(a))] = 1.0                   # solo el ganador refuerza
        elif self.rule == "softmax":
            g = np.maximum(np.tanh((a - a.mean()) / max(self.temperature, 1e-6)), 0.0)
        else:  # 'above_mean'
            g = np.tanh(np.maximum(a - a.mean(), 0.0) / (a.std() + 1e-8))
        return g

    # ------------------------------------------------------- inhibicion lateral local
    def configure_inhibition(self, *, spacing: int = 5, offset: int | None = None, radius: int = 8,
                             metric: str = "cheby", fire_threshold: float = 0.40, K: float = 0.10,
                             gain: float = 1.0, mode: str = "fraction") -> int:
        """Coloca neuronas inhibidoras SUPERPUESTAS en una rejilla regular del mapa (cada `spacing`)
        y precalcula, para cada una, su REGION asignada: las neuronas dentro de `radius` (metrica
        `cheby`=cuadrado por defecto). Devuelve el nº de inhibidores. Activa la inhibicion."""
        if offset is None:
            offset = spacing // 2
        self.inhib_on = True
        self.inhib_spacing, self.inhib_offset, self.inhib_radius = spacing, offset, radius
        self.inhib_metric, self.fire_threshold = metric, fire_threshold
        self.inhib_K, self.inhib_gain, self.inhib_mode = K, gain, mode
        centers = [(r, c) for r in range(offset, self.grid_h, spacing)
                          for c in range(offset, self.grid_w, spacing)]
        self._inhib_regions = []
        for rc, cc in centers:
            dr, dc = np.abs(self._nr - rc), np.abs(self._nc - cc)
            if metric == "euclid":
                mask = (dr.astype(np.int64) ** 2 + dc.astype(np.int64) ** 2) <= radius * radius
            elif metric == "manhattan":
                mask = (dr + dc) <= radius
            else:  # 'cheby' -> cuadrado
                mask = (dr <= radius) & (dc <= radius)
            self._inhib_regions.append(np.nonzero(mask)[0])
        return len(self._inhib_regions)

    def _inhibition_coeffs(self, a: np.ndarray) -> np.ndarray:
        """Coeficiente s_i >= 0 a RESTAR del empuje de cada neurona DISPARADA, por la inhibicion
        lateral de todos los inhibidores que la cubren. Cada inhibidor mira la fraccion de sus
        neuronas que disparan (activacion >= fire_threshold); si supera K, reduce; si no, nada.

        La ganancia inhibidora es INDEPENDIENTE del lr: la reduccion es `inhib_gain * exceso` (no se
        escala con el learning rate), asi se controla la balanza refuerzo/inhibicion por separado."""
        fired = a >= self.fire_threshold
        s = np.zeros(self.n_out, dtype=np.float32)
        for idx in self._inhib_regions:
            fr = fired[idx]
            nf = int(fr.sum())
            if nf == 0:
                continue
            base = nf if self.inhib_mode == "hinge" else nf / idx.size
            diff = base - self.inhib_K
            if self.inhib_mode == "sigmoid":
                e = float(np.log1p(np.exp(10.0 * diff)) / 10.0)   # softplus: ~0 por debajo de K
            else:                                                  # 'fraction' | 'hinge'
                e = diff if diff > 0 else 0.0
            if e > 0:
                s[idx[fr]] += self.inhib_gain * e                # independiente del lr
        return s

    # ------------------------------------------------------------------ un paso online
    def learn_sample(self, x: np.ndarray, lr: float) -> int:
        """Presenta x. El algoritmo base SOLO refuerza (incrementa) los pesos de las neuronas cerca
        de disparar. Las neuronas inhibidoras son las UNICAS que reducen pesos (a las disparadas de
        regiones con exceso de disparos). Refuerzo e inhibicion actuan sobre el mismo eje `xu`, asi
        que se combinan en un solo coeficiente por neurona: coef_i = lr*g_i - s_i."""
        xu = self._normalize_vec(x)
        a = self.activate(xu)
        winner = int(np.argmax(a))
        self.win_count[winner] += 1

        coef = lr * self.reinforce_gain * self._gate(a)    # refuerzo: lr * ganancia_activacion * g
        if self.inhib_on:
            coef = coef - self._inhibition_coeffs(a)       # inhibicion: gain independiente del lr
        idx = np.nonzero(coef)[0]
        if idx.size:
            self.W[idx] += coef[idx][:, None] * xu[None, :]
            self.W[idx] = self._normalize_rows(self.W[idx])
        return winner

    def learn_epoch(self, X: np.ndarray, lr: float, rng: np.random.Generator,
                    *, on_frame=None, frame_every: int = 0) -> dict:
        """Una PASADA (iteracion) sobre todo el dataset en orden aleatorio. Devuelve metricas.

        Si `on_frame` y `frame_every` estan dados, llama `on_frame()` cada `frame_every` muestras
        (para capturar la animacion densa del proceso DENTRO de la epoca, no solo al final)."""
        order = rng.permutation(len(X))
        wins_before = self.win_count.copy()
        for k, i in enumerate(order):
            self.learn_sample(X[i], lr)
            if on_frame is not None and frame_every and (k + 1) % frame_every == 0:
                on_frame()
        self.epochs_trained += 1
        # metricas de la epoca
        A = self.activate_batch(self._normalize_rows(X))   # X ya normalizado por fila
        winners = A.argmax(axis=1)
        max_act = A.max(axis=1)
        epoch_wins = self.win_count - wins_before
        used = np.count_nonzero(epoch_wins)
        fired = (A >= self.fire_threshold).sum(axis=1)   # neuronas que disparan por entrada
        return {
            "mean_winner_activation": float(max_act.mean()),
            "coverage": used / self.n_out,          # fraccion de neuronas que ganaron algo esta epoca
            "unique_winners": int(len(np.unique(winners))),
            "dead_units": int(self.n_out - np.count_nonzero(self.win_count)),
            "mean_fired": float(fired.mean()),      # media de neuronas disparadas por entrada (theta)
        }

    # ------------------------------------------------------------------ vistas para graficar
    def receptive_fields(self, img_h: int, img_w: int) -> np.ndarray:
        """Pesos de cada neurona como imagen (n_out, img_h, img_w)."""
        return self.W.reshape(self.n_out, img_h, img_w)

    # ------------------------------------------------------------------ persistencia
    def save(self, path: str | Path) -> None:
        """Guarda TODO el estado (pesos + hiperparametros + contadores) para poder reanudar.

        Guarda tambien `epochs_trained`: nº total de iteraciones acumuladas, para llevar la cuenta
        aunque se reanude con otro set de entradas."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W=self.W,
            win_count=self.win_count,
            epochs_trained=np.int64(self.epochs_trained),
            n_in=np.int64(self.n_in),
            n_out=np.int64(self.n_out),
            rule=np.str_(self.rule),
            temperature=np.float64(self.temperature),
            anti=np.float64(self.anti),
            reinforce_gain=np.float64(self.reinforce_gain),
            grid_h=np.int64(self.grid_h),
            grid_w=np.int64(self.grid_w),
            inhib_on=np.int64(1 if self.inhib_on else 0),
            inhib_spacing=np.int64(self.inhib_spacing),
            inhib_offset=np.int64(self.inhib_offset),
            inhib_radius=np.int64(self.inhib_radius),
            inhib_metric=np.str_(self.inhib_metric),
            fire_threshold=np.float64(self.fire_threshold),
            inhib_K=np.float64(self.inhib_K),
            inhib_gain=np.float64(self.inhib_gain),
            inhib_mode=np.str_(self.inhib_mode),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompetitiveLayer":
        """Reconstruye una capa desde un archivo .npz de `save()`, lista para seguir entrenando.
        Compatible con modelos antiguos (sin campos de geometria/inhibicion): usa los defaults."""
        d = np.load(Path(path), allow_pickle=False)
        keys = set(d.files)

        def g(k, default, cast):
            return cast(d[k]) if k in keys else default

        layer = cls(
            int(d["n_in"]), int(d["n_out"]), rule=str(d["rule"]),
            temperature=float(d["temperature"]), anti=float(d["anti"]),
            reinforce_gain=g("reinforce_gain", 1.0, float),
            grid_h=g("grid_h", None, int), grid_w=g("grid_w", None, int),
            inhib_on=bool(g("inhib_on", 0, int)),
            inhib_spacing=g("inhib_spacing", 5, int), inhib_offset=g("inhib_offset", None, int),
            inhib_radius=g("inhib_radius", 8, int), inhib_metric=g("inhib_metric", "cheby", str),
            fire_threshold=g("fire_threshold", 0.40, float), inhib_K=g("inhib_K", 0.10, float),
            inhib_gain=g("inhib_gain", 1.0, float), inhib_mode=g("inhib_mode", "fraction", str),
        )
        layer.W = d["W"].astype(np.float32)
        layer.win_count = d["win_count"].astype(np.int64)
        layer.epochs_trained = int(d["epochs_trained"])
        return layer
