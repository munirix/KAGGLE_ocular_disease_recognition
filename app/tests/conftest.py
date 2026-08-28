"""conftest.py -- fixtures e configuracao global dos testes."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Garante que os imports de src funcionem quando pytest e executado de app/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as C  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Executa testes de integracao lentos"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marca testes lentos de integracao")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="use --run-slow para executar testes de integracao")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# Diretorios de saida redirecionados durante os testes de integracao.
_REDIRECTED_DIRS = (
    "OUTPUTS_DIR", "FIGURES_DIR", "MODELS_DIR", "PREDICTIONS_DIR",
    "CACHE_DIR", "REPORTS_DIR", "PREPROCESSED_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def isolate_outputs(request, tmp_path, monkeypatch):
    """Isola as saidas dos testes lentos em um diretorio temporario.

    Os testes de integracao executam o pipeline de ponta a ponta e, sem este
    isolamento, sobrescreveriam os artefatos da execucao de referencia
    (metricas, figuras, modelos e predicoes citados na monografia). Redirecionar
    os diretorios de saida preserva a reprodutibilidade do experimento
    documentado, mantendo o cache de leitura dos dados de entrada intacto.
    """
    if "slow" not in request.keywords:
        yield
        return

    sandbox = tmp_path / "outputs"
    for name in _REDIRECTED_DIRS:
        target = sandbox if name == "OUTPUTS_DIR" else sandbox / name.replace("_DIR", "").lower()
        monkeypatch.setattr(C, name, target, raising=True)
    monkeypatch.setattr(C, "FEATURES_CACHE", sandbox / "cache" / C.FEATURES_CACHE.name, raising=True)
    C.ensure_dirs()
    yield