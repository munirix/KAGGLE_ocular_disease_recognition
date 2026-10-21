"""
src/predict.py -- inferência em uma nova imagem de fundo de olho
================================================================
Módulo reutilizável para predição sobre uma única imagem, usado tanto por
``app/predict.py`` (CLI standalone) quanto por ``main.py predict``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from . import features, preprocessing
from .logging_utils import LOGGER, setup_file_logging
from .utils import save_json

_SEVERITY_NAMES = {i: name for i, name in enumerate(C.SEVERITY_CLASSES)}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_stage2_threshold(default: float = 0.5) -> float:
    """Recupera o limiar de decisão do melhor modelo da Etapa 2."""
    metrics_path = C.REPORTS_DIR / "stage2_metrics.json"
    try:
        data = _load_json(metrics_path)
        return float(data["operating_point"]["threshold"])
    except Exception:
        LOGGER.warning("Nao foi possivel ler o limiar de %s; usando %.3f", metrics_path, default)
        return default


def predict_image(path: str | Path, stages: tuple[int, ...] = (2, 3, 4)) -> dict:
    """Pré-processa a imagem, extrai features e aplica os modelos treinados."""
    C.ensure_dirs()
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {img_path}")

    LOGGER.info("Predicao para %s", img_path)

    # Etapa 1 e extracao de features.
    proc = preprocessing.preprocess_image(img_path)
    x, _ = features.extract_features(proc)
    x = x.reshape(1, -1)

    out = {
        "filename": str(img_path.name),
        "warning": "Etapa 2 e um classificador experimental; Etapas 3 e 4 usam alvos sinteticos.",
    }

    # Etapa 2: Normal x Miopia.
    if 2 in stages:
        stage2_model_path = C.MODELS_DIR / "stage2_best_model.joblib"
        if not stage2_model_path.exists():
            raise FileNotFoundError(f"Modelo da Etapa 2 nao encontrado: {stage2_model_path}")

        model = joblib.load(stage2_model_path)
        score = float(model.predict_proba(x)[0, 1])
        threshold = _get_stage2_threshold()
        pred = int(score >= threshold)
        out["stage2"] = {
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "prediction": C.BINARY_NAMES[pred],
            "prediction_int": pred,
        }

        # Etapas 3 e 4 so fazem sentido se houver predicao de miopia.
        if pred == 1:
            if 3 in stages:
                stage3_model_path = C.MODELS_DIR / "stage3_best_model.joblib"
                if stage3_model_path.exists():
                    s3 = joblib.load(stage3_model_path)
                    sev_int = int(np.round(s3.predict(x)[0]))
                    sev_int = max(0, min(sev_int, len(C.SEVERITY_CLASSES) - 1))
                    out["stage3"] = {"severity": _SEVERITY_NAMES[sev_int]}
                else:
                    out["stage3"] = {"error": f"Modelo nao encontrado: {stage3_model_path}"}

            if 4 in stages:
                stage4_model_path = C.MODELS_DIR / "stage4_best_model.joblib"
                if stage4_model_path.exists():
                    s4 = joblib.load(stage4_model_path)
                    diopter = float(s4.predict(x)[0])
                    out["stage4"] = {"diopter_d": round(diopter, 2)}
                else:
                    out["stage4"] = {"error": f"Modelo nao encontrado: {stage4_model_path}"}
        else:
            out["stage3"] = {"skipped": "imagem classificada como normal"}
            out["stage4"] = {"skipped": "imagem classificada como normal"}

    return out


def run_predict(
    cfg: C.RunConfig,
    image: str | Path,
    stages: tuple[int, ...] = (2, 3, 4),
    output: Path | None = None,
) -> dict:
    """Entry-point usado pela CLI unificada ``main.py predict``.

    ``cfg`` garante logging, semente e diretórios consistentes com as demais
    etapas do pipeline.
    """
    C.ensure_dirs()
    if cfg.log_file:
        setup_file_logging(Path(cfg.log_file))

    result = predict_image(image, stages=stages)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(result, output)
    return result