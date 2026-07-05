"""Bitácora ÚNICA y versionada de entrenamientos: `trainings/TRAININGS.md`.

Registra cada entrenamiento (hecho o EN CURSO) en una tabla Markdown, para revisar de un vistazo
qué hay entrenándose y qué se completó. Se mantiene AUTOMÁTICAMENTE desde los scripts:
`log_training(id=..., estado="en_curso"|"hecho"|..., ...)` hace *upsert* de la fila por `id`
(solo actualiza los campos que se pasan; el resto se conserva). Es la fuente de verdad y el
archivo legible a la vez (no hay sidecar).

Pensado para llamarse muchas veces por corrida (al iniciar, por época y al terminar). Un `Lock`
serializa las escrituras dentro de un proceso; no está pensado para escritores concurrentes en
procesos distintos (poco probable aquí, donde cada corrida es un proceso secuencial)."""
from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

LOG_PATH = Path("trainings") / "TRAININGS.md"

# columnas de la tabla (la 1ª, "id", es la clave de upsert)
_COLS = ["id", "modelo", "datos", "estado", "épocas", "val", "test", "checkpoint", "actualizado"]
_TITLE = ("# Bitácora de entrenamientos\n\n"
          "Registro automático (lo escriben los scripts de entrenamiento). Fila por corrida, "
          "clave = `id`. Estados: `en_curso`, `hecho`, `fallido`.\n\n")
_LEGEND = ("\n> Generado automáticamente por `nnist.utils.trainlog`. No editar a mano mientras "
           "haya corridas activas (se reescribe entero en cada actualización).\n")
_lock = Lock()


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).replace("|", "/").strip() or "—"


def _is_sep(cell: str) -> bool:
    return set(cell.strip()) <= {"-", ":"} and "-" in cell


def _parse(path: Path):
    """Lee la tabla existente -> (dict id->fila, orden de ids). Tolera que el archivo no exista."""
    rows: dict[str, dict] = {}
    order: list[str] = []
    if not path.exists():
        return rows, order
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(_COLS):
            continue
        if cells[0] == "id" or _is_sep(cells[0]):     # cabecera o separador
            continue
        row = dict(zip(_COLS, cells))
        rows[cells[0]] = row
        order.append(cells[0])
    return rows, order


def _write(path: Path, rows: dict, order: list) -> None:
    widths = {c: len(c) for c in _COLS}
    for i in order:
        for c in _COLS:
            widths[c] = max(widths[c], len(rows[i][c]))

    def render(cells: dict) -> str:
        return "| " + " | ".join(cells[c].ljust(widths[c]) for c in _COLS) + " |"

    header = render({c: c for c in _COLS})
    sep = render({c: "-" * widths[c] for c in _COLS})
    body = "\n".join(render(rows[i]) for i in order)
    text = _TITLE + header + "\n" + sep + ("\n" + body if body else "") + "\n" + _LEGEND
    path.write_text(text, encoding="utf-8")


def log_training(id: str, *, modelo=None, datos=None, estado=None, épocas=None,
                 val=None, test=None, checkpoint=None, path: Path | str = LOG_PATH) -> None:
    """Upsert de la fila `id` en la bitácora. Solo se actualizan los campos != None."""
    path = Path(path)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows, order = _parse(path)
        row = rows.get(id) or {c: "—" for c in _COLS}
        row["id"] = str(id)
        for key, value in (("modelo", modelo), ("datos", datos), ("estado", estado),
                           ("épocas", épocas), ("val", val), ("test", test),
                           ("checkpoint", checkpoint)):
            if value is not None:
                row[key] = _fmt(value)
        row["actualizado"] = time.strftime("%Y-%m-%d %H:%M")
        if id not in rows:
            order.append(id)
        rows[id] = row
        _write(path, rows, order)
