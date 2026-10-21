"""test_predict.py -- testes da inferencia em uma nova imagem."""
from __future__ import annotations

from pathlib import Path

import pytest

from src import config as C
from src.predict import predict_image, run_predict


def test_predict_image_not_found():
    with pytest.raises(FileNotFoundError):
        predict_image("caminho/inexistente.jpg")


def test_run_predict_missing_model(tmp_path, monkeypatch):
    """predict deve levantar FileNotFoundError quando o modelo da Etapa 2 nao existe."""
    import cv2
    import numpy as np

    # Cria uma imagem RGB valida para passar pelo pre-processamento.
    dummy = tmp_path / "dummy.jpg"
    rng = np.random.default_rng(42)
    cv2.imwrite(str(dummy), rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))

    # Garante que nao ha modelo da Etapa 2 no diretorio de modelos.
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        run_predict(C.RunConfig(), image=str(dummy), stages=(2, 3, 4))


@pytest.mark.slow
def test_predict_end_to_end(tmp_path):
    """Teste de integracao: executa o pipeline de smoke e prediz uma imagem real."""
    import main

    ret = main.main(["all", "--max-per-class", "5", "--no-tune"])
    assert ret == 0

    model_path = C.MODELS_DIR / "stage2_best_model.joblib"
    assert model_path.exists(), f"Modelo nao gerado: {model_path}"

    # Usa uma imagem de miopia conhecida (233_right.jpg) ou fallback.
    img = C.PROVIDED_PREPROCESSED_DIR / "233_right.jpg"
    if not img.exists():
        pytest.skip("Imagem de teste nao encontrada")

    output = tmp_path / "result.json"
    ret = main.main(["predict", str(img), "--stages", "2", "3", "4", "-o", str(output)])
    assert ret == 0
    assert output.exists()
    import json
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["filename"] == img.name
    assert "stage2" in result
    assert result["stage2"]["prediction"] in C.BINARY_NAMES