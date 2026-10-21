"""
main.py -- ponto de entrada (CLI) do pipeline do TCC
====================================================
Exemplos de uso (a partir da pasta app/):

    python main.py all                       # pipeline completo (Etapas 1-4 + EDA)
    python main.py eda                       # apenas analise exploratoria
    python main.py stage1                    # pre-processamento + antes/depois
    python main.py stage2                    # classificacao Normal x Miopia
    python main.py stage3                    # severidade (leve/moderado/alta/magna)
    python main.py stage4                    # regressao do grau (dioptrias)
    python main.py ablation                  # ablacao de atributos na Etapa 4
    python main.py predict imagem.jpg        # inferencia em nova imagem

    python main.py all --max-per-class 150   # smoke test rapido (subamostra)
    python main.py all --force               # ignora caches e recomputa tudo
    python main.py predict imagem.jpg --stages 2 3 4 -o saida.json

Saidas: figuras em outputs/figures, metricas em outputs/reports, modelos em
outputs/models.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src import ablation_study
from src import config as C
from src import pipeline
from src.predict import run_predict
from src.logging_utils import LOGGER, setup_tee_logging


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Argumentos compartilhados pelos comandos do pipeline."""
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limita amostras por classe (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Ignora caches de pre-processamento/features.")
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument("--jobs", type=int, default=C.N_JOBS,
                        help="Nº de processos paralelos (-1 = todos os nucleos).")
    parser.add_argument("--no-tune", action="store_true",
                        help="Desativa a busca de hiperparametros (HalvingGridSearchCV) na Etapa 2.")
    parser.add_argument("--target-sensitivity", type=float,
                        default=C.STAGE2_TARGET_SENSITIVITY,
                        help="Sensibilidade-alvo (recall da miopia) para escolher o limiar na Etapa 2.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Caminho do log de execucao. Padrao: outputs/execution_YYYYMMDD_HHMMSS.log")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de triagem do rotulo ocular de miopia em imagens de fundo de olho (ODIR-5K)."
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                help="Etapa a executar.")

    # Comandos do pipeline (compartilham argumentos comuns).
    pipeline_commands = {
        "eda": "Analise exploratoria",
        "stage1": "Pre-processamento + visualizacao antes/depois",
        "stage2": "Classificacao Normal x Miopia",
        "stage3": "Severidade (leve/moderado/alta/magna)",
        "stage4": "Regressao do grau (dioptrias)",
        "ablation": "Ablacao de atributos na Etapa 4",
        "all": "Pipeline completo (Etapas 1-4 + EDA)",
    }
    for cmd, help_text in pipeline_commands.items():
        sub_cmd = sub.add_parser(cmd, help=help_text)
        _add_common_args(sub_cmd)

    # Comando predict (argumentos proprios).
    predict_cmd = sub.add_parser("predict", help="Inferencia em uma nova retinografia.")
    predict_cmd.add_argument("image", type=str, help="Caminho para a imagem JPEG/PNG.")
    predict_cmd.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4],
                             help="Etapas a executar (padrao: 2 3 4).")
    predict_cmd.add_argument("--output", "-o", type=Path, default=None,
                             help="Caminho para salvar o JSON de saida.")
    predict_cmd.add_argument("--log-file", type=str, default=None,
                             help="Caminho do log de execucao.")

    return parser


def _run_pipeline_command(command: str, args: argparse.Namespace, cfg: C.RunConfig) -> None:
    commands = {
        "eda": pipeline.run_eda,
        "stage1": pipeline.run_stage1,
        "stage2": pipeline.run_stage2,
        "stage3": pipeline.run_stage3,
        "stage4": pipeline.run_stage4,
        "ablation": ablation_study.run_ablation,
        "all": pipeline.run_all,
    }
    LOGGER.info("Comando=%s | log_file=%s | cfg=%s", command, args.log_file, cfg)
    commands[command](cfg)
    LOGGER.info("Concluido: %s", command)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    C.ensure_dirs()

    # Define e ativa o log de execucao (tee) antes de qualquer outra operacao,
    # garantindo que stdout, stderr e os registros do LOGGER fiquem versionados.
    if getattr(args, "log_file", None):
        log_path = Path(args.log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = C.OUTPUTS_DIR / f"execution_{timestamp}.log"
    setup_tee_logging(log_path)

    if args.command == "predict":
        LOGGER.info("Comando=%s | image=%s | output=%s | log=%s",
                    args.command, args.image, args.output, log_path)
        run_predict(
            C.RunConfig(
                max_per_class=None,
                force_recompute=False,
                seed=C.RANDOM_SEED,
                n_jobs=C.N_JOBS,
                log_file=str(log_path),
                stage2_tune=C.STAGE2_TUNE,
                stage2_target_sensitivity=C.STAGE2_TARGET_SENSITIVITY,
            ),
            image=args.image,
            stages=tuple(args.stages),
            output=args.output,
        )
        LOGGER.info("Concluido: %s", args.command)
        return 0

    cfg = C.RunConfig(
        max_per_class=getattr(args, "max_per_class", None),
        force_recompute=getattr(args, "force", False),
        seed=getattr(args, "seed", C.RANDOM_SEED),
        n_jobs=getattr(args, "jobs", C.N_JOBS),
        log_file=str(log_path),
        stage2_tune=not getattr(args, "no_tune", False),
        stage2_target_sensitivity=getattr(args, "target_sensitivity", C.STAGE2_TARGET_SENSITIVITY),
    )
    _run_pipeline_command(args.command, args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())