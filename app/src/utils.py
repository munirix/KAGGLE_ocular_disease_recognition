"""
utils.py
========
Utilidades transversais: logging, semente global, serializacao de metricas e um
mapeador paralelo com barra de progresso. Mante-las separadas evita duplicacao
e deixa os modulos de dominio (pre-processamento, modelos) focados na sua logica.
"""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "miopia") -> logging.Logger:
    """Retorna um logger configurado uma unica vez (idempotente)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = get_logger()


def set_global_seed(seed: int) -> None:
    """Fixa as sementes de random e numpy (scikit-learn usa random_state proprio)."""
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timer(label: str):
    """Mede e loga o tempo de um bloco de codigo (diagnostico de gargalos)."""
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
    """
    Aplica `func` a cada item em paralelo (joblib) com barra de progresso (tqdm).

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