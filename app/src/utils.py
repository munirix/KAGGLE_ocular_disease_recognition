"""
utils.py
========
Utilidades de execucao: semente global (determinismo), medicao de tempo,
serializacao de metricas e mapeador paralelo com barra de progresso.

Responsabilidades vizinhas ficam em modulos proprios para evitar um modulo
"saco de gatos": registro de eventos em `logging_utils.py` e metadados de
proveniencia em `provenance.py`.
"""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .logging_utils import LOGGER


def set_global_seed(seed: int) -> None:
    """Fixa as sementes de random e numpy (scikit-learn usa random_state proprio)."""
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timer(label: str):
    """Mede e loga o tempo de um bloco de codigo (diagnostico de gargalos).

    A duracao registrada aqui e persistida no log de execucao pelo `Tee`,
    permitindo auditar o custo de cada etapa a posteriori.
    """
    start = time.perf_counter()
    LOGGER.info("[inicio] %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.info("[fim]    %s (%.2fs)", label, elapsed)


def save_json(obj: Any, path: Path) -> None:
    """Serializa metricas em JSON, convertendo tipos numpy nao serializaveis."""
    def _default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_default)
    LOGGER.info("Metricas salvas em %s", path)


def parallel_map(
    func: Callable,
    items: Sequence,
    n_jobs: int = -1,
    desc: str = "",
    use_threads: bool = False,
) -> list:
    """Aplica `func` a cada item em paralelo (joblib) com barra de progresso (tqdm).

    Usamos processos por padrao (CPU-bound: OpenCV/numpy). Para IO puro,
    `use_threads=True` evita o custo de serializacao entre processos.
    """
    from joblib import Parallel, delayed
    try:
        from tqdm import tqdm
        iterator: Iterable = tqdm(items, desc=desc, unit="img")
    except Exception:  # tqdm e opcional
        iterator = items

    backend = "threading" if use_threads else "loky"
    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(func)(x) for x in iterator
    )