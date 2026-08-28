"""test_features.py -- invariantes do vetor de atributos.

Estes testes fixam o contrato numerico da engenharia de atributos: a dimensao
declarada na monografia (169), a composicao por bloco, o determinismo e a
ausencia de valores nao finitos. Uma alteracao acidental no extrator quebra
aqui, antes de contaminar os resultados das etapas de modelagem.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src.features import (_build_uniform_lbp_table, _color_features,
                          _gradient_features, _radial_features,
                          _texture_features, extract_features)

EXPECTED_TOTAL = 169
EXPECTED_BLOCKS = {"cor": 91, "textura": 60, "gradiente": 12, "radial": 6}


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Imagem RGB sintetica deterministica no tamanho de saida do pre-processamento."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8)


def test_total_feature_count(rgb_image):
    feats, names = extract_features(rgb_image)
    assert len(feats) == EXPECTED_TOTAL
    assert len(names) == EXPECTED_TOTAL


def test_block_composition(rgb_image):
    """Cada bloco deve produzir exatamente a quantidade documentada no Apendice A."""
    import cv2
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    mask = gray > C.FUNDUS_CROP_THRESHOLD

    blocks = {
        "cor": _color_features(rgb_image, mask),
        "textura": _texture_features(gray, mask),
        "gradiente": _gradient_features(gray, mask),
        "radial": _radial_features(rgb_image, gray, mask),
    }
    observed = {name: len(values) for name, (values, _) in blocks.items()}
    assert observed == EXPECTED_BLOCKS
    assert sum(observed.values()) == EXPECTED_TOTAL

    # Valores e nomes devem estar alinhados em cada bloco.
    for name, (values, labels) in blocks.items():
        assert len(values) == len(labels), f"bloco {name} desalinhado"


def test_lbp_uniform_table_has_59_bins():
    """LBP com P=8 produz 58 padroes uniformes mais um bin nao-uniforme."""
    _, n_bins = _build_uniform_lbp_table(C.LBP_P)
    assert n_bins == 59


def test_feature_names_are_unique(rgb_image):
    _, names = extract_features(rgb_image)
    assert len(set(names)) == len(names)


def test_features_are_finite(rgb_image):
    feats, _ = extract_features(rgb_image)
    assert np.all(np.isfinite(feats)), "atributos nao devem conter NaN nem inf"


def test_extraction_is_deterministic(rgb_image):
    first, names_a = extract_features(rgb_image)
    second, names_b = extract_features(rgb_image)
    np.testing.assert_array_equal(first, second)
    assert names_a == names_b


def test_constant_image_does_not_break_extraction():
    """Canal constante torna a assimetria indefinida; o extrator deve degradar bem."""
    flat = np.full((C.IMAGE_SIZE, C.IMAGE_SIZE, 3), 128, dtype=np.uint8)
    feats, names = extract_features(flat)
    assert len(feats) == EXPECTED_TOTAL
    assert np.all(np.isfinite(feats))


def test_histograms_are_normalised(rgb_image):
    """Histogramas de cor e de matiz somam aproximadamente 1 por canal."""
    feats, names = extract_features(rgb_image)
    lookup = dict(zip(names, feats))
    for prefix, n_bins in (("hist_r_", C.COLOR_HIST_BINS),
                           ("hist_g_", C.COLOR_HIST_BINS),
                           ("hist_b_", C.COLOR_HIST_BINS),
                           ("hist_hue_", C.HUE_HIST_BINS)):
        total = sum(lookup[f"{prefix}{i}"] for i in range(n_bins))
        assert total == pytest.approx(1.0, abs=1e-5), f"{prefix} nao normalizado"