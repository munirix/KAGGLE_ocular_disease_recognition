"""
main.py  -- ponto de entrada (CLI) do pipeline do TCC
=====================================================
Exemplos de uso (a partir da pasta app/):

    python main.py all                 # pipeline completo (Etapas 1-4 + EDA)
    python main.py eda                 # apenas analise exploratoria
    python main.py stage1              # pre-processamento + antes/depois
    python main.py stage2              # classificacao Normal x Miopia
    python main.py stage3              # severidade (leve/alta/magna)
    python main.py stage4              # regressao do grau (dioptrias)

    python main.py all --max-per-class 150  # smoke test rapido (subamostra)
    python main.py all --force              # ignora caches e recomputa tudo

Saidas: figuras em outputs/figures, metricas em outputs/reports, modelos em
outputs/models.
"""
from __future__ import annotations

import argparse
import sys

from src import config as C
from src import pipeline
from src.utils import LOGGER

_COMMANDS = {
    "eda": pipeline.run_eda,
    "stage1": pipeline.run_stage1,
    "stage2": pipeline.run_stage2,
    "stage3": pipeline.run_stage3,
    "stage4": pipeline.run_stage4,
    "all": pipeline.run_all,
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de diagnostico de miopia em imagens de fundo de olho (ODIR-5K)."
    )
    parser.add_argument("command", choices=sorted(_COMMANDS),
                        help="Etapa a executar.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limita amostras por classe (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Ignora caches de pre-processamento/features.")
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument("--jobs", type=int, default=C.N_JOBS,
                        help="Nº de processos paralelos (-1 = todos os nucleos).")
    parser.add_argument("--use-provided-preprocessed", action="store_true",
                        help="Usa as imagens ja pre-processadas do dataset.")
    parser.add_argument("--no-tune", action="store_true",
                        help="Desativa a busca de hiperparametros (HalvingGridSearchCV) na Etapa 2.")
    parser.add_argument("--target-sensitivity", type=float,
                        default=C.STAGE2_TARGET_SENSITIVITY,
                        help="Sensibilidade-alvo (recall da miopia) para escolher o limiar na Etapa 2.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = C.RunConfig(
        max_per_class=args.max_per_class,
        force_recompute=args.force,
        seed=args.seed,
        n_jobs=args.jobs,
        use_provided_preprocessed=args.use_provided_preprocessed,
        stage2_tune=not args.no_tune,
        stage2_target_sensitivity=args.target_sensitivity,
    )
    LOGGER.info("Comando=%s | cfg=%s", args.command, cfg)
    _COMMANDS[args.command](cfg)
    LOGGER.info("Concluido: %s", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())