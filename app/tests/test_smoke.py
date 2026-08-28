"""test_smoke.py -- testes de fumaca do pipeline completo e de etapas isoladas."""
from __future__ import annotations

import pytest

import main
from src import config as C


@pytest.mark.slow
class TestSmoke:
    """Executam o pipeline em subamostras pequenas; marcados como slow."""

    def test_stage2_smoke(self):
        ret = main.main(["stage2", "--max-per-class", "5", "--no-tune"])
        assert ret == 0
        assert (C.REPORTS_DIR / "stage2_metrics.json").exists()
        assert (C.MODELS_DIR / "stage2_best_model.joblib").exists()

    def test_stage1_smoke(self):
        ret = main.main(["stage1", "--max-per-class", "5"])
        assert ret == 0
        assert (C.REPORTS_DIR / "stage1_results.txt").exists()

    def test_eda_smoke(self):
        ret = main.main(["eda", "--max-per-class", "5"])
        assert ret == 0
        assert (C.FIGURES_DIR / "eda_01_labels_8classes.png").exists()