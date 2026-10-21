"""
logging_utils.py
================
Observabilidade da execucao: logger estruturado, handler de arquivo e
redirecionamento `Tee` de stdout/stderr.

Este modulo concentra tudo o que diz respeito a *registro de eventos*. Os
utilitarios de determinismo, serializacao e paralelismo ficam em `utils.py`;
a coleta de metadados de execucao fica em `provenance.py`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOGGER_NAME = "miopia"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
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


def setup_file_logging(path: Path) -> None:
    """Adiciona um FileHandler ao LOGGER global (idempotente)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(path):
            return
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    LOGGER.addHandler(fh)


class Tee:
    """Duplica uma stream de texto (stdout/stderr) para o terminal e para arquivo.

    Equivale ao comando Unix `tee`: tudo que e escrito na stream aparece tanto no
    console quanto no arquivo de execucao. Isso torna o log versionado por run,
    sem exigir redirecionamento manual pelo usuario.
    """

    def __init__(self, stream, path: Path, mode: str = "a", encoding: str = "utf-8"):
        self.terminal = stream
        self.file = open(path, mode, encoding=encoding)
        self.path = path

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def isatty(self) -> bool:
        return getattr(self.terminal, "isatty", lambda: False)()

    def getvalue(self) -> str:
        """Compatibilidade com pytest/StringIO: le o conteudo escrito ate agora."""
        self.file.flush()
        try:
            return self.path.read_text(encoding="utf-8")
        except Exception:
            return ""


def setup_tee_logging(path: Path) -> None:
    """Redireciona stdout e stderr para um Tee que tambem escreve em `path`.

    Reconfigura os handlers de stream do LOGGER para que as mensagens geradas
    pela biblioteca de logging tambem passem pelo Tee, garantindo que tudo
    (stdout, stderr e registros estruturados) fique no arquivo de execucao.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, path)
    sys.stderr = Tee(sys.__stderr__, path)
    # Faz handlers de stream do LOGGER apontarem para o novo stderr (Tee).
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.stream = sys.stderr