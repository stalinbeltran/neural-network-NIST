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

Regla de aprendizaje (Hebbiana CON SIGNO, graduada):
  Para cada neurona se calcula un "gate" g_i con SIGNO segun su cercania a disparar:
    - g_i > 0  para las neuronas cerca de dispararse (activacion alta)  -> REFUERZAN.
    - g_i < 0  para las neuronas lejos de dispararse (activacion baja)  -> DEBILITAN.
  y se actualiza cada peso en proporcion a la entrada que estuvo presente:

    Delta w_ij = lr * g_i * x_j

  Es decir: si una entrada x_j estuvo activa, el peso hacia una neurona CERCA de dispararse se
  fortalece (mas probable que dispare la proxima vez) y el peso hacia una neurona LEJOS de
  dispararse se debilita (menos probable). Tras cada muestra se renormaliza w_i a norma 1 (mantiene
  la activacion en escala coseno y evita que los pesos exploten).

  El gate g_i se reparte de forma balanceada e independiente del nº de neuronas: la parte positiva
  suma 1 (refuerzo total) y la negativa suma `anti` (castigo total, `anti=1` -> equilibrado).

  Variantes (--rule):
    - 'above_mean' (def): g_raw = a - media(a). Encima de la media refuerza, debajo debilita.
    - 'softmax'         : g_raw = softmax(a/T) - 1/n (competicion suave global, tambien con signo).
    - 'wta'             : la mas activa refuerza; TODAS las demas debilitan un poco (anti-Hebbiano).

Implementado en numpy: rapido y sin dependencias de framework.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class CompetitiveLayer:
    def __init__(self, n_in: int, n_out: int, *, rule: str = "above_mean",
                 temperature: float = 0.1, anti: float = 1.0, seed: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.rule = rule
        self.temperature = temperature
        self.anti = anti                       # fuerza del castigo anti-Hebbiano (neuronas lejanas)
        rng = np.random.default_rng(seed)
        # pesos aleatorios pequenos, luego normalizados por fila a norma 1
        W = rng.standard_normal((n_out, n_in)).astype(np.float32)
        self.W = self._normalize_rows(W)
        # contador de "victorias" por neurona (para diagnostico: cobertura, neuronas muertas)
        self.win_count = np.zeros(n_out, dtype=np.int64)
        self.epochs_trained = 0                # iteraciones acumuladas (persiste entre reanudaciones)

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

    # ------------------------------------------------------------------ gate con signo
    def _gate(self, a: np.ndarray) -> np.ndarray:
        """Gate g_i con SIGNO en [-1, 1], POR NEURONA (no diluido entre todas):
          g_i > 0  neurona cerca de disparar  -> refuerza sus pesos hacia la entrada presente.
          g_i < 0  neurona lejos de disparar   -> debilita sus pesos hacia la entrada presente.
        La magnitud crece con lo lejos que esta la activacion de la media. `anti` escala el lado
        negativo (castigo anti-Hebbiano): anti=1 simetrico, anti=0 desactiva el debilitamiento."""
        if self.rule == "wta":
            g = np.full_like(a, -1.0)
            g[int(np.argmax(a))] = 1.0                   # solo el ganador refuerza; el resto debilita
        elif self.rule == "softmax":
            g = np.tanh((a - a.mean()) / max(self.temperature, 1e-6))
        else:  # 'above_mean'
            g = np.tanh((a - a.mean()) / (a.std() + 1e-8))
        return np.where(g < 0.0, self.anti * g, g)

    # ------------------------------------------------------------------ un paso online
    def learn_sample(self, x: np.ndarray, lr: float) -> int:
        """Presenta x: refuerza pesos de neuronas cercanas a disparar y debilita los de las lejanas,
        en ambos casos EN PROPORCION a la entrada presente (Delta w_ij = lr*g_i*x_j)."""
        xu = self._normalize_vec(x)
        a = self.activate(xu)
        winner = int(np.argmax(a))
        self.win_count[winner] += 1

        g = self._gate(a)
        idx = np.nonzero(g)[0]              # neuronas con empuje (refuerzo o castigo) no nulo
        if idx.size:
            self.W[idx] += lr * g[idx][:, None] * xu[None, :]
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
        return {
            "mean_winner_activation": float(max_act.mean()),
            "coverage": used / self.n_out,          # fraccion de neuronas que ganaron algo esta epoca
            "unique_winners": int(len(np.unique(winners))),
            "dead_units": int(self.n_out - np.count_nonzero(self.win_count)),
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
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompetitiveLayer":
        """Reconstruye una capa desde un archivo .npz de `save()`, lista para seguir entrenando."""
        d = np.load(Path(path), allow_pickle=False)
        layer = cls(int(d["n_in"]), int(d["n_out"]), rule=str(d["rule"]),
                    temperature=float(d["temperature"]), anti=float(d["anti"]))
        layer.W = d["W"].astype(np.float32)
        layer.win_count = d["win_count"].astype(np.int64)
        layer.epochs_trained = int(d["epochs_trained"])
        return layer
