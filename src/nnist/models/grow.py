"""Crecimiento gradual de una CNN (Net2Net, Chen et al. 2015).

Idea: empezar con una red pequeña (barata de entrenar) e ir AÑADIÉNDOLE capacidad —más
canales por bloque (ancho) y/o más bloques conv (profundidad)— reutilizando lo ya aprendido.

`grow_cnn(old, channels=..., fc_hidden=...)` construye una `SimpleCNN` mayor y le trasplanta
los pesos de `old`:

- **Ensanchar (Net2Wider):** para pasar un ancho w -> w', se REPLICAN filtros existentes
  (elegidos al azar) en los canales nuevos y, en la capa CONSUMIDORA, se dividen los pesos
  entrantes de cada unidad replicada por su nº de copias. Resultado: la red grande calcula
  EXACTAMENTE la misma función que la pequeña en el instante del trasplante (cero bache de
  accuracy). Esto vale para ensanchar bloques conv y para ensanchar la capa densa oculta.

- **Profundizar (añadir bloques):** cada bloque de `SimpleCNN` incluye un `MaxPool(2)`, así que
  un bloque nuevo cambia el tamaño espacial y la dimensión de aplanado -> NO se puede preservar la
  función exactamente (decisión D2 del proyecto: se acepta un pequeño bache al profundizar). Se
  conserva ("warm-start") el prefijo de bloques compartido y la cabeza densa arranca de init fresco.

Regla: `channels`/`fc_hidden` nuevos deben ser >= los viejos en el prefijo compartido (la red
solo CRECE). Ensanchar sin cambiar la profundidad es exacto; profundizar es warm-start.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from .cnn import SimpleCNN


def _conv_blocks(model: SimpleCNN) -> list[dict]:
    """Extrae los bloques conv de `model.features` como [{conv, bn}] (bn=None si no hay)."""
    blocks: list[dict] = []
    cur: dict | None = None
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            if cur is not None:
                blocks.append(cur)
            cur = {"conv": layer, "bn": None}
        elif isinstance(layer, nn.BatchNorm2d):
            assert cur is not None
            cur["bn"] = layer
    if cur is not None:
        blocks.append(cur)
    return blocks


def _linears(model: SimpleCNN) -> list[nn.Linear]:
    """Capas Linear de la cabeza densa, en orden: [oculta, salida] o solo [salida]."""
    return [l for l in model.classifier if isinstance(l, nn.Linear)]


def _widen_mapping(old_w: int, new_w: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """Mapa Net2Wider g: [0,new_w) -> [0,old_w). g[j]=j para j<old_w; el resto se replica al azar.
    Devuelve (g, cnt) donde cnt[i] = nº de veces que la unidad vieja i aparece en g."""
    if new_w < old_w:
        raise ValueError(f"Net2Wider solo ensancha: new={new_w} < old={old_w}")
    g = list(range(old_w)) + [rng.randrange(old_w) for _ in range(new_w - old_w)]
    cnt = [0] * old_w
    for i in g:
        cnt[i] += 1
    return g, cnt


@torch.no_grad()
def _copy_conv(dst: nn.Conv2d, src: nn.Conv2d, g_out: list[int],
               g_in: list[int] | None, cnt_in: list[int] | None) -> None:
    """Trasplanta pesos de conv `src` a `dst` replicando salidas por `g_out` y, si se ensanchó el
    bloque anterior, expandiendo entradas por `g_in` con división por `cnt_in` (Net2Wider)."""
    w = src.weight.data                       # (out_old, in_old, k, k)
    w = w[g_out]                              # replica canales de SALIDA -> (out_new, in_old, k, k)
    if g_in is not None:
        w = w[:, g_in]                       # expande canales de ENTRADA -> (out_new, in_new, k, k)
        div = torch.tensor([cnt_in[i] for i in g_in], dtype=w.dtype).view(1, -1, 1, 1)
        w = w / div                          # divide por nº de copias del canal de entrada
    dst.weight.data.copy_(w)
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data[g_out])


@torch.no_grad()
def _copy_bn(dst: nn.BatchNorm2d, src: nn.BatchNorm2d, g: list[int]) -> None:
    """Replica los parámetros y estadísticas de BatchNorm por el mapa `g`."""
    dst.weight.data.copy_(src.weight.data[g])
    dst.bias.data.copy_(src.bias.data[g])
    dst.running_mean.data.copy_(src.running_mean.data[g])
    dst.running_var.data.copy_(src.running_var.data[g])


@torch.no_grad()
def _transfer(old: SimpleCNN, new: SimpleCNN, seed: int) -> None:
    """Trasplanta pesos de `old` a `new` (misma input_shape/num_classes). Exacto si la profundidad
    no cambia (solo se ensancha); warm-start del prefijo compartido si se añaden bloques."""
    rng = random.Random(seed)
    ob, nb = _conv_blocks(old), _conv_blocks(new)
    depth_equal = len(ob) == len(nb)
    shared = min(len(ob), len(nb))

    # mapa de ensanchado por bloque compartido (g_in del bloque i = g_out del bloque i-1)
    gmaps: list[tuple[list[int], list[int]]] = []
    prev_g = prev_cnt = None
    for i in range(shared):
        old_w = ob[i]["conv"].out_channels
        new_w = nb[i]["conv"].out_channels
        g, cnt = _widen_mapping(old_w, new_w, rng)
        _copy_conv(nb[i]["conv"], ob[i]["conv"], g_out=g, g_in=prev_g, cnt_in=prev_cnt)
        if ob[i]["bn"] is not None and nb[i]["bn"] is not None:
            _copy_bn(nb[i]["bn"], ob[i]["bn"], g)
        gmaps.append((g, cnt))
        prev_g, prev_cnt = g, cnt

    # Cabeza densa: solo se puede trasplantar exacto si NO cambió la profundidad (n_flat estable).
    # Si se profundizó, los bloques nuevos y la cabeza quedan con su init fresco (warm-start / D2).
    if not depth_equal:
        return

    g_last, cnt_last = gmaps[-1]
    old_lins, new_lins = _linears(old), _linears(new)
    # dimensión espacial (H*W) tras los bloques: n_flat / nº_canales_último_bloque
    hw_old = old_lins[0].in_features // ob[-1]["conv"].out_channels

    # 1) expandir la ENTRADA de la 1ª Linear por los canales replicados del último bloque conv.
    #    El aplanado es channel-major: el bloque de columnas del canal c es [c*hw : (c+1)*hw).
    w1 = old_lins[0].weight.data                       # (fc_old, n_flat_old)
    cols = []
    for j in g_last:
        block = w1[:, j * hw_old:(j + 1) * hw_old] / cnt_last[j]
        cols.append(block)
    w1_in = torch.cat(cols, dim=1)                     # (fc_old, n_flat_new)

    if len(old_lins) == 1:                              # cabeza sin capa oculta: solo salida
        new_lins[0].weight.data.copy_(w1_in)
        new_lins[0].bias.data.copy_(old_lins[0].bias.data)
        return

    # 2) ensanchar las unidades OCULTAS (filas de la 1ª Linear) por Net2Wider, y dividir la
    #    entrada de la 2ª Linear (salida) por el nº de copias de cada unidad oculta.
    old_fc = old_lins[0].out_features
    new_fc = new_lins[0].out_features
    gh, cnth = _widen_mapping(old_fc, new_fc, rng)
    new_lins[0].weight.data.copy_(w1_in[gh])
    new_lins[0].bias.data.copy_(old_lins[0].bias.data[gh])

    w2 = old_lins[1].weight.data                       # (num_classes, fc_old)
    div = torch.tensor([cnth[gh[j]] for j in range(new_fc)], dtype=w2.dtype)
    new_lins[1].weight.data.copy_(w2[:, gh] / div)
    new_lins[1].bias.data.copy_(old_lins[1].bias.data)


def grow_cnn(old: SimpleCNN, *, channels=None, kernel_size=None, fc_hidden=None,
             batchnorm=None, dropout=None, seed: int = 0) -> SimpleCNN:
    """Devuelve una `SimpleCNN` mayor que `old` con sus pesos trasplantados (Net2Net).

    Cualquier hiperparámetro no indicado hereda el de `old`. `channels`/`fc_hidden` deben CRECER
    respecto a `old` (la red solo se agranda). Ensanchar con la misma profundidad preserva la
    función exactamente; añadir bloques es warm-start (ver módulo)."""
    new = SimpleCNN(
        input_shape=old.input_shape,
        num_classes=old.num_classes,
        channels=old.channels if channels is None else tuple(channels),
        kernel_size=old.kernel_size if kernel_size is None else kernel_size,
        fc_hidden=old.fc_hidden if fc_hidden is None else fc_hidden,
        batchnorm=old.batchnorm if batchnorm is None else batchnorm,
        dropout=old.dropout if dropout is None else dropout,
    )
    if new.kernel_size != old.kernel_size or new.batchnorm != old.batchnorm:
        raise ValueError("grow_cnn no cambia kernel_size ni batchnorm (romperían el trasplante).")
    _transfer(old, new, seed)
    return new
