# neural-network-NIST

Reconocimiento de caracteres del set **NIST** con redes neuronales en **PyTorch**.

El proyecto está diseñado para comparar **estrategias de red** distintas bajo dos ejes:

- **Entrada completa**: la red ve toda la matriz de píxeles del dígito.
- **Entrada por subsets**: la red ve solo un recorte/ventana de la matriz.

Para cada estrategia hay un sistema de **barridos de hiperparámetros** y un registro **multi-métrica**
(accuracy, número de parámetros, coste, tamaño del área de entrada...) para optimizar según el criterio
que interese en cada caso.

> Las reglas, decisiones de diseño y convenciones del proyecto están en [CLAUDE.md](CLAUDE.md).

## Estado

Pipeline funcional end-to-end para **MLP** sobre MNIST, en estrategia **completa** y **subset**:
loader (MNIST/EMNIST) + transform de recorte + `Trainer` + `runner` + `sweep` + registro
multi-métrica. Verificado (1 época): completa (1×28×28, 50 890 params) 91,7% vs. subset
(1×14×14, 13 258 params) 88,3%.

Pendiente (`TODO` en el código): `SimpleCNN`, loader de NIST SD19 crudo, y callbacks
(early stopping / checkpoint por época).

## Setup

Se detectó Python 3.14.6 en la máquina, pero PyTorch puede no tener wheels estables para 3.14 aún.
Se recomienda un venv con **Python 3.12**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

## Uso rápido

```powershell
python scripts/download_data.py --dataset mnist
python scripts/train.py  --config configs/models/mlp_full.yaml
python scripts/sweep.py  --config configs/sweeps/mlp_full_grid.yaml
```

Ver [CLAUDE.md](CLAUDE.md) para la estructura completa de archivos y el flujo de trabajo.
