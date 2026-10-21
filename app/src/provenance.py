"""
provenance.py
=============
Proveniencia e reprodutibilidade da execucao: hash de arquivos de dados,
commit do Git, inventario de versoes de pacotes, consumo de memoria de pico e
construcao do manifesto persistido por cada etapa.

Separado de `utils.py` porque responde a uma pergunta distinta: nao "como
executar", mas "sob quais condicoes este resultado foi produzido".
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Pacotes cujas versoes entram no manifesto. Inclui as dependencias de
# modelagem (mord sustenta a regressao ordinal da Etapa 3) e de teste (pytest),
# para que o manifesto descreva o ambiente completo e nao apenas o numerico.
_TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "mord", "matplotlib",
    "seaborn", "opencv-python-headless", "joblib", "tqdm", "openpyxl", "pytest",
)


def file_sha256(path: Path, chunk: int = 8192) -> str:
    """Hash SHA-256 de um arquivo para rastreabilidade de dados."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def get_git_commit(path: Path | None = None) -> str | None:
    """Retorna o hash curto do commit atual, se estiver em um repo git."""
    try:
        cwd = path if path is not None else Path.cwd()
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def peak_memory_mb() -> float | None:
    """Memoria residente de pico do processo, em MiB.

    Usa `resource.getrusage`, cuja unidade de `ru_maxrss` depende da plataforma:
    bytes no macOS e kilobytes no Linux. Retorna None onde o modulo `resource`
    nao esta disponivel (por exemplo, Windows).
    """
    try:
        import resource
    except ImportError:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(raw / divisor, 1)


def get_environment_info() -> dict[str, Any]:
    """Coleta informacoes minimas do ambiente de execucao para o manifesto."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": get_git_commit(),
        "package_versions": versions,
    }


def build_manifest(cfg: Any, command: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Constroi um manifesto de execucao com config, ambiente, caches e metadados extras."""
    from . import config as C
    manifest = get_environment_info()
    manifest["command"] = command
    manifest["config"] = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    manifest["cache"] = {
        "preprocessed_dir": str(C.PREPROCESSED_CACHE_DIR.name),
        "features_cache": str(C.FEATURES_CACHE.name),
    }
    manifest["peak_memory_mb"] = peak_memory_mb()
    if extra:
        manifest.update(extra)
    return manifest