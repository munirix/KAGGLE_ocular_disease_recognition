"""test_preprocessing.py -- contrato geometrico do pre-processamento e falhas de leitura.

Cobre o comportamento diante de imagens corrompidas ou ausentes, um risco de
falha silenciosa levantado na analise de qualidade de engenharia.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src.preprocessing import (apply_clahe, crop_fundus, normalize01,
                               pad_to_square, preprocess_image, read_rgb,
                               resize_square)


@pytest.fixture
def sample_jpeg(tmp_path):
    """Grava uma imagem JPEG valida, com moldura preta, para exercitar o recorte."""
    import cv2
    rng = np.random.default_rng(7)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[50:250, 100:300] = rng.integers(60, 255, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "fundo.jpg"
    cv2.imwrite(str(path), img)
    return path


def test_preprocess_output_contract(sample_jpeg):
    """A saida deve ser sempre uint8 no formato quadrado configurado."""
    out = preprocess_image(sample_jpeg)
    assert out.shape == (C.IMAGE_SIZE, C.IMAGE_SIZE, 3)
    assert out.dtype == np.uint8


def test_preprocess_return_steps_exposes_stages(sample_jpeg):
    """Os cinco estagios usados nas figuras da Etapa 1 devem estar disponiveis."""
    steps = preprocess_image(sample_jpeg, return_steps=True)
    assert isinstance(steps, dict)
    assert list(steps) == ["1_original", "2_cropped", "3_resized", "4_clahe", "5_normalized"]
    # A versao normalizada existe apenas para visualizacao, nao para modelagem.
    assert steps["5_normalized"].max() <= 1.0
    assert steps["4_clahe"].dtype == np.uint8


def test_crop_fundus_removes_black_border(sample_jpeg):
    original = read_rgb(sample_jpeg)
    cropped = crop_fundus(original)
    assert cropped.shape[0] < original.shape[0]
    assert cropped.shape[1] < original.shape[1]


def test_crop_fundus_empty_mask_returns_original():
    """Mascara vazia (imagem toda escura) deve devolver a imagem intacta."""
    black = np.zeros((64, 48, 3), dtype=np.uint8)
    result = crop_fundus(black)
    assert result.shape == black.shape


def test_pad_to_square_and_resize():
    rect = np.full((40, 90, 3), 200, dtype=np.uint8)
    squared = pad_to_square(rect)
    assert squared.shape[0] == squared.shape[1] == 90
    resized = resize_square(squared)
    assert resized.shape == (C.IMAGE_SIZE, C.IMAGE_SIZE, 3)


def test_apply_clahe_preserves_shape_and_dtype():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8)
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_normalize01_range():
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    norm = normalize01(img)
    assert norm.min() >= 0.0 and norm.max() <= 1.0


def test_read_rgb_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        read_rgb(tmp_path / "nao_existe.jpg")


def test_read_rgb_corrupted_file_raises(tmp_path):
    """Arquivo com extensao de imagem mas conteudo invalido deve falhar de forma explicita."""
    corrupted = tmp_path / "corrompida.jpg"
    corrupted.write_bytes(b"isto nao e uma imagem JPEG")
    with pytest.raises(Exception):
        read_rgb(corrupted)


def test_preprocess_image_corrupted_file_raises(tmp_path):
    corrupted = tmp_path / "corrompida.png"
    corrupted.write_bytes(b"\x89PNG\r\n\x1a\n" + b"lixo")
    with pytest.raises(Exception):
        preprocess_image(corrupted)