# CLAUDE.md — Condiciones y guía del proyecto

Reconocimiento de caracteres sobre el set **NIST** con redes neuronales en **Python + PyTorch**.
Este archivo define las reglas, decisiones y convenciones que Claude debe respetar al trabajar
en este repositorio. Léelo antes de implementar o modificar cualquier cosa.

---

## 1. Objetivo del proyecto

Construir y comparar distintas **estrategias de red neuronal** para clasificar imágenes de
caracteres del set NIST, con dos ejes de experimentación:

1. **Entrada completa**: la red recibe *todos* los píxeles de la matriz de la imagen del dígito.
2. **Entrada por subsets**: la red recibe solo un **área más pequeña** de la matriz (un recorte /
   ventana / región de la imagen). Se quiere estudiar cuánto rendimiento se conserva viendo menos.

Para **cada** tipo/estructura/estrategia de red debe existir un sistema que permita **entrenarla con
distintos conjuntos de parámetros** (barridos de hiperparámetros) y **optimizar su rendimiento**.

## 2. Definición de "rendimiento" (multi-objetivo)

El rendimiento **no** es una sola métrica. El sistema debe medir y registrar, como mínimo:

- **Precisión** (accuracy) y otras métricas de clasificación (precision/recall/F1 por clase, matriz de confusión).
- **Número de parámetros** del modelo (`count_params()` — total y entrenables).
- **Tamaño del área de entrada** usada (para la estrategia de subsets: forma y ubicación del recorte).
- **Coste**: tiempo de entrenamiento, tiempo de inferencia, memoria/FLOPs si es viable.

Cada corrida debe emitir un **registro de resultados** comparable entre modelos y barridos, para poder
elegir según el criterio que importe en cada momento (p. ej. "máxima accuracy" vs. "mejor accuracy por parámetro").

## 3. Decisiones ya tomadas (no re-litigar sin preguntar)

| Decisión | Valor |
|---|---|
| Lenguaje | Python |
| Framework | **PyTorch** |
| Alcance de clases | Empezar con **dígitos 0–9 (10 clases)**, pero el diseño **no** debe hardcodear el nº de clases; debe generalizar a letras (EMNIST / NIST SD19, 36 o 62 clases). |
| Fuente de datos inicial | **MNIST / EMNIST** (descarga automática vía `torchvision`), detrás de una **interfaz de dataset abstracta** que luego admita **NIST SD19** crudo. |

## 4. Entorno

- Python detectado en la máquina: **3.14.6**. PyTorch/torchvision pueden **no** tener wheels estables
  para 3.14 todavía → se recomienda un **venv con Python 3.12** para instalar PyTorch sin fricción.
  Antes de asumir que algo "no funciona", verifica la versión del intérprete del venv activo.
- Todo se ejecuta desde la raíz del repo. Usa el paquete `src/nnist` (instalable en editable: `pip install -e .`).

## 5. Principios de diseño (reglas para Claude)

1. **Todo experimento se define por configuración, no por código hardcodeado.** Un experimento =
   un archivo YAML en `configs/` que referencia un modelo, un dataset, unas transformaciones y unos
   hiperparámetros. Cambiar de estrategia no debe requerir editar el código del `Trainer`.
2. **Registries por nombre.** Modelos, datasets y transformaciones se registran por string
   (`@register("mlp")`) para poder seleccionarlos desde el YAML. Añadir una arquitectura nueva =
   un archivo nuevo en `models/` + su registro, **sin tocar** el runner ni el sweep.
3. **La estrategia "subset" es una transformación de datos, no un modelo aparte.** El recorte/ventana
   de la matriz vive en `data/transforms.py`. El modelo solo debe adaptarse a la **forma de entrada**
   resultante (que se deduce de la config, no se hardcodea).
4. **Separar los tres ejes**: (a) *qué ve la red* (transform/subset), (b) *qué estructura tiene la red*
   (model), (c) *cómo se entrena* (hiperparámetros/trainer). Un barrido puede variar cualquiera de los tres.
5. **Reproducibilidad**: fijar semillas, guardar la config exacta + métricas + checkpoint de cada corrida
   en `experiments/<run_id>/`. Un resultado sin su config asociada no sirve.
6. **El conteo de parámetros es un ciudadano de primera clase.** `BaseModel` expone `count_params()`;
   se registra siempre en las métricas.
7. **No metas datasets, checkpoints ni logs en git.** Ver `.gitignore` (`data/raw`, `experiments/`, etc.).

## 6. Estructura de archivos

```
neural-network-NIST/
├── CLAUDE.md                 # este archivo
├── README.md
├── requirements.txt
├── pyproject.toml            # paquete instalable `nnist`
│
├── configs/                  # DEFINICIÓN de experimentos (YAML), sin lógica
│   ├── base.yaml             # defaults compartidos
│   ├── models/               # una config por estrategia de red
│   │   ├── mlp_full.yaml      #   MLP sobre la imagen completa
│   │   ├── cnn_full.yaml      #   CNN sobre la imagen completa
│   │   └── mlp_subset.yaml    #   MLP sobre un subset/ventana de la imagen
│   └── sweeps/               # barridos de hiperparámetros / param-sets
│       └── mlp_full_grid.yaml
│
├── data/
│   ├── raw/                  # datasets descargados (gitignored)
│   └── processed/            # datos preprocesados (gitignored)
│
├── src/nnist/                # PAQUETE — toda la lógica reutilizable
│   ├── data/
│   │   ├── datasets.py       # loaders detrás de interfaz común (MNIST/EMNIST -> luego SD19)
│   │   ├── transforms.py     # transforms, incl. recorte/ventana (estrategia SUBSET)
│   │   └── registry.py
│   ├── models/
│   │   ├── base.py           # BaseModel + count_params()
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   └── registry.py
│   ├── training/
│   │   ├── trainer.py        # bucle de entrenamiento/validación agnóstico al modelo
│   │   └── callbacks.py      # early stopping, checkpointing, etc.
│   ├── evaluation/
│   │   └── metrics.py        # accuracy, nº params, coste... registro multi-métrica
│   ├── experiments/
│   │   ├── config.py         # carga/validación de configs (dataclasses)
│   │   ├── runner.py         # ejecuta UNA corrida a partir de una config
│   │   └── sweep.py          # expande y ejecuta un BARRIDO de configs
│   └── utils/
│       ├── seed.py
│       └── logging.py
│
├── scripts/                  # ENTRADAS de línea de comandos (delgadas)
│   ├── download_data.py
│   ├── train.py              #  python scripts/train.py --config configs/models/mlp_full.yaml
│   └── sweep.py              #  python scripts/sweep.py --config configs/sweeps/mlp_full_grid.yaml
│
├── experiments/              # SALIDAS: <run_id>/{config.yaml, metrics.json, model.pt, log} (gitignored)
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_metrics.py
```

### Regla mnemotécnica de dónde va cada cosa
- **`configs/`** = *qué* experimento correr (datos + declarativo).
- **`src/nnist/`** = *cómo* funciona (lógica reutilizable, con tests).
- **`scripts/`** = *lanzar* algo (CLI delgada, sin lógica de negocio).
- **`experiments/`** = *qué salió* (resultados reproducibles).

## 7. Flujo de trabajo típico

```bash
# 1. Preparar entorno (venv con Python 3.12 recomendado)
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt

# 2. Descargar datos
python scripts/download_data.py --dataset mnist

# 3. Entrenar una estrategia con una config
python scripts/train.py --config configs/models/mlp_full.yaml

# 4. Barrer hiperparámetros para optimizar
python scripts/sweep.py --config configs/sweeps/mlp_full_grid.yaml

# 5. Comparar resultados de experiments/ (accuracy vs nº params, etc.)
```

## 8. Cómo añadir cosas (para Claude)

- **Nueva arquitectura**: archivo en `src/nnist/models/`, heredar de `BaseModel`, `@register("nombre")`,
  y una config en `configs/models/`. No tocar `runner`/`sweep`.
- **Nueva estrategia de subset**: añadir un transform en `data/transforms.py` (registrado), y referenciarlo
  desde la config. El modelo se adapta por la forma de entrada declarada en la config.
- **Nueva métrica**: añadir en `evaluation/metrics.py`; debe integrarse en el registro de resultados
  para que sea comparable en los barridos.

## 9. Convenciones

- Tests con `pytest` en `tests/`. Toda lógica en `src/nnist` debe ser testeable sin GPU y sin descargar
  el dataset completo (usar tensores/fixtures pequeños).
- Nada de rutas absolutas dentro del código; resolver paths relativos a la raíz del repo.
- Formato de `run_id`: `<modelo>_<estrategia>_<timestamp>` para trazabilidad.
