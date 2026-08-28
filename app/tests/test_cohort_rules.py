"""test_cohort_rules.py -- regra de rotulagem da coorte e calibracao do limiar.

Fixa duas decisoes metodologicas centrais da monografia: a assimetria entre a
classe negativa (igualdade exata) e a positiva (ocorrencia de termo), e a regra
de escolha do limiar por sensibilidade-alvo com fallback de Youden.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src.data_loader import _classify_row, _eye_of
from src.stage2_classification import _select_threshold, _wilson_interval

# -----------------------------------------------------------------------------
# Regra de rotulagem
# -----------------------------------------------------------------------------

def test_normal_requires_exact_match():
    assert _classify_row("normal fundus") == C.CLASS_NORMAL
    # Texto que apenas contem "normal fundus" nao e negativo estrito.
    assert _classify_row("lens dust, normal fundus") != C.CLASS_NORMAL


def test_myopia_uses_substring_match():
    assert _classify_row("pathological myopia") == C.CLASS_MYOPIA
    assert _classify_row("myopic maculopathy") == C.CLASS_MYOPIA
    # Combinacao com outro achado permanece positiva (comorbidade possivel).
    assert _classify_row("pathological myopia, drusen") == C.CLASS_MYOPIA


def test_unrelated_keyword_is_excluded():
    assert _classify_row("cataract") is None
    assert _classify_row("moderate non proliferative retinopathy") is None


def test_classification_is_case_insensitive():
    assert _classify_row("Normal Fundus") == C.CLASS_NORMAL
    assert _classify_row("PATHOLOGICAL MYOPIA") == C.CLASS_MYOPIA


def test_myopia_keywords_count_matches_documentation():
    """A monografia declara seis fragmentos de texto para a classe positiva."""
    assert len(C.MYOPIA_KEYWORDS) == 6


def test_eye_of_filename():
    assert _eye_of("1520_left.jpg") == "left"
    assert _eye_of("233_right.jpg") == "right"


# -----------------------------------------------------------------------------
# Calibracao do limiar
# -----------------------------------------------------------------------------

def test_selected_threshold_meets_sensitivity_target():
    """Com escores separaveis, o limiar escolhido deve atingir a sensibilidade-alvo."""
    y = np.array([0] * 40 + [1] * 10)
    scores = np.concatenate([np.linspace(0.0, 0.4, 40), np.linspace(0.6, 0.99, 10)])
    thr, rule = _select_threshold(y, scores, target_sensitivity=0.90)
    achieved = ((scores >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert achieved >= 0.90
    assert "sensibilidade" in rule


def test_threshold_falls_back_to_youden_when_target_unreachable():
    """Alvo impossivel (>1) forca o fallback documentado para o indice de Youden."""
    y = np.array([0] * 20 + [1] * 5)
    rng = np.random.default_rng(0)
    scores = rng.random(25)
    _, rule = _select_threshold(y, scores, target_sensitivity=1.5)
    assert "youden" in rule.lower()


def test_lower_threshold_increases_sensitivity():
    """Reduzir o alvo nao pode exigir um limiar mais permissivo do que um alvo maior."""
    y = np.array([0] * 30 + [1] * 10)
    scores = np.concatenate([np.linspace(0.0, 0.5, 30), np.linspace(0.4, 0.95, 10)])
    thr_low, _ = _select_threshold(y, scores, target_sensitivity=0.50)
    thr_high, _ = _select_threshold(y, scores, target_sensitivity=0.95)
    assert thr_high <= thr_low


# -----------------------------------------------------------------------------
# Intervalo de Wilson usado no relato de incerteza
# -----------------------------------------------------------------------------

def test_wilson_interval_brackets_point_estimate():
    lo, hi = _wilson_interval(36, 48)
    assert lo < 36 / 48 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_handles_zero_sample():
    assert _wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_with_more_data():
    narrow = _wilson_interval(500, 1000)
    wide = _wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])