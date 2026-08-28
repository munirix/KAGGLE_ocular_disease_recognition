"""
predict.py -- CLI standalone de inferência em uma nova imagem de fundo de olho
==============================================================================
Wrapper de conveniência ao redor de ``src.predict``. Permite executar:

    .venv/bin/python predict.py /caminho/para/imagem.jpg
    .venv/bin/python predict.py /caminho/para/imagem.jpg -o saida.json
    .venv/bin/python predict.py /caminho/para/imagem.jpg --stages 2 3 4

Para uso integrado com o pipeline, preferir ``python main.py predict``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permite executar o script diretamente de app/ sem depender do PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src import config as C
from src.predict import predict_image
from src.logging_utils import LOGGER, setup_file_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inferencia para uma nova retinografia.")
    parser.add_argument("image", help="Caminho para a imagem JPEG/PNG.")
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4],
                        help="Etapas a executar (padrao: 2 3 4).")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Caminho para salvar o JSON de saida.")
    parser.add_argument("--log", type=Path, default=C.OUTPUTS_DIR / "predict.log",
                        help="Caminho do arquivo de log.")
    args = parser.parse_args(argv)

    C.ensure_dirs()
    setup_file_logging(args.log)

    try:
        result = predict_image(args.image, stages=tuple(args.stages))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    except Exception as exc:
        LOGGER.exception("Falha na predicao")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())