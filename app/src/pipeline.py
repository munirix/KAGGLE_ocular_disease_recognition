"""
pipeline.py
===========
Orquestra as etapas do TCC. Cada funcao publica pode ser chamada isoladamente
(via main.py) ou em conjunto por `run_all`. Como o pre-processamento e as
features sao cacheados em disco, executar uma etapa isolada reaproveita o
trabalho ja feito pelas anteriores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C
from . import data_loader, features, labeling, preprocessing
from . import stage2_classification as s2
from . import stage3_severity as s3
from . import stage4_regression as s4
from . import visualization as viz
from .logging_utils import LOGGER, setup_file_logging
from .utils import save_json, set_global_seed, timer


# -----------------------------------------------------------------------------
# Preparacao compartilhada (coorte + matriz de features), com cache
# -----------------------------------------------------------------------------

def prepare(cfg: C.RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    C.ensure_dirs()
    setup_file_logging(C.OUTPUTS_DIR / "pipeline.log")
    set_global_seed(cfg.seed)
    full_df = data_loader.load_full_df()
    cohort = data_loader.build_cohort(cfg)
    X, names = features.build_feature_matrix(cohort, cfg)
    return full_df, cohort, X, names


def prepare_myopia(cfg: C.RunConfig, cohort: pd.DataFrame,
                   X: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Recorta a coorte de miopia (alinhando X), divide em treino/teste por paciente
    e gera os alvos sinteticos usando APENAS estatisticas do treino."""
    mask = (cohort["label"] == 1).to_numpy()
    X_my = X[mask]
    myopia_df = data_loader.myopia_subset(cohort)         # mesma ordem de `mask`

    # Split por paciente ANTES de gerar os rotulos sinteticos.
    groups = myopia_df[C.GROUP_COL].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=cfg.seed)
    train_idx, test_idx = next(gss.split(X_my, groups=groups))

    # Alvos derivados do treino; teste sofre a mesma transformacao (sem vazar estatisticas).
    myopia_df = labeling.assign_synthetic_targets(myopia_df, cfg, fit_idx=train_idx)
    assert len(myopia_df) == X_my.shape[0], "Desalinhamento X_my x myopia_df"
    return X_my, myopia_df, train_idx, test_idx


# -----------------------------------------------------------------------------
# Etapa 0 - Analise exploratoria
# -----------------------------------------------------------------------------

def run_eda(cfg: C.RunConfig | None = None) -> None:
    cfg = cfg or C.RunConfig()
    full_df, cohort, X, _ = prepare(cfg)
    with timer("EDA"):
        viz.plot_label_distribution_full(full_df)
        viz.plot_binary_distribution(cohort)
        viz.plot_demographics(cohort)
        _, myopia_df, _, _ = prepare_myopia(cfg, cohort, X)
        viz.plot_severity_distribution(myopia_df)
        viz.plot_diopter_distribution(myopia_df)


# -----------------------------------------------------------------------------
# Etapa 1 - Pre-processamento (+ visualizacao antes x depois)
# -----------------------------------------------------------------------------

def _sample_rows(cohort: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for cls in ("N", "M"):
        sub = cohort[cohort["class_str"] == cls]
        if len(sub):
            parts.append(sub.sample(min(per_class, len(sub)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def run_stage1(cfg: C.RunConfig | None = None) -> None:
    cfg = cfg or C.RunConfig()
    _, cohort, _, _ = prepare(cfg)
    with timer("Etapa 1 - pre-processamento"):
        preprocessing.preprocess_cohort(cohort, cfg)

    # Visualizacoes antes x depois.
    samples = _sample_rows(cohort, per_class=2, seed=cfg.seed)
    pairs = []
    sample_figs = []
    for _, row in samples.iterrows():
        steps = preprocessing.preprocess_image(row["path"], return_steps=True)
        label = "Miopia" if row["class_str"] == "M" else "Normal"
        fig_name = f"stage1_steps_{row['class_str']}_{row['filename'].replace('.jpg', '')}.png"
        viz.plot_preprocessing_steps(
            steps, f"Etapa 1 - {label} ({row['filename']})", fig_name
        )
        sample_figs.append(fig_name)
        pairs.append((steps["1_original"], steps["4_clahe"], label))
    viz.plot_original_vs_preprocessed(pairs)

    # Resumo legivel da Etapa 1
    n_preproc = len(list(C.PREPROCESSED_CACHE_DIR.glob("*.png")))
    n_normal = int((cohort["label"] == C.BINARY_LABELS[C.CLASS_NORMAL]).sum())
    n_miopia = int((cohort["label"] == C.BINARY_LABELS[C.CLASS_MYOPIA]).sum())
    n_patients = int(cohort[C.GROUP_COL].nunique())
    lines = [
        "=== Resultados da Etapa 1 (Pre-processamento) ===",
        f"Imagens pre-processadas: {n_preproc}",
        f"Dimensao final: {C.IMAGE_SIZE}x{C.IMAGE_SIZE}",
        f"Recorte do fundo: limiar {C.FUNDUS_CROP_THRESHOLD}",
        f"CLAHE: clip limit {float(C.CLAHE_CLIP_LIMIT)}, tile grid {C.CLAHE_TILE_GRID}",
        f"Cache: {C.PREPROCESSED_CACHE_DIR.name}",
        f"Coorte: {n_normal} Normal / {n_miopia} Miopia ({n_patients} pacientes)",
        f"Amostras visualizadas: {len(samples)} imagens ({', '.join(sample_figs)})",
        "Figuras geradas:",
        "  - viz_original_vs_preproc.png",
    ]
    for name in sample_figs:
        lines.append(f"  - {name}")
    results_path = C.REPORTS_DIR / "stage1_results.txt"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Resultados da Etapa 1 salvos em %s", results_path)


# -----------------------------------------------------------------------------
# Etapas 2, 3 e 4
# -----------------------------------------------------------------------------

def run_stage2(cfg: C.RunConfig | None = None, command: str = "stage2") -> dict:
    cfg = cfg or C.RunConfig()
    _, cohort, X, names = prepare(cfg)
    return s2.run_stage2(X, cohort, names, cfg, command=command)


def run_stage3(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    _, cohort, X, names = prepare(cfg)
    X_my, myopia_df, train_idx, test_idx = prepare_myopia(cfg, cohort, X)
    return s3.run_stage3(X_my, myopia_df, names, cfg,
                         train_idx=train_idx, test_idx=test_idx, command="stage3")


def run_stage4(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    _, cohort, X, names = prepare(cfg)
    X_my, myopia_df, train_idx, test_idx = prepare_myopia(cfg, cohort, X)
    return s4.run_stage4(X_my, myopia_df, names, cfg,
                         train_idx=train_idx, test_idx=test_idx, command="stage4")


# -----------------------------------------------------------------------------
# Pipeline completo
# -----------------------------------------------------------------------------

def run_all(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    LOGGER.info("################### PIPELINE COMPLETO ###################")
    full_df, cohort, X, names = prepare(cfg)

    # Etapa 1 (cache + figuras).
    preprocessing.preprocess_cohort(cohort, cfg)
    samples = _sample_rows(cohort, per_class=2, seed=cfg.seed)
    pairs = []
    for _, row in samples.iterrows():
        steps = preprocessing.preprocess_image(row["path"], return_steps=True)
        label = "Miopia" if row["class_str"] == "M" else "Normal"
        viz.plot_preprocessing_steps(
            steps, f"Etapa 1 - {label} ({row['filename']})",
            f"stage1_steps_{row['class_str']}_{row['filename'].replace('.jpg', '')}.png"
        )
        pairs.append((steps["1_original"], steps["4_clahe"], label))
    viz.plot_original_vs_preprocessed(pairs)

    # EDA.
    viz.plot_label_distribution_full(full_df)
    viz.plot_binary_distribution(cohort)
    viz.plot_demographics(cohort)

    # Etapa 2.
    r2 = s2.run_stage2(X, cohort, names, cfg, command="all")

    # Etapas 3 e 4 (coorte de miopia + alvos sinteticos).
    X_my, myopia_df, train_idx, test_idx = prepare_myopia(cfg, cohort, X)
    viz.plot_severity_distribution(myopia_df)
    viz.plot_diopter_distribution(myopia_df)
    r3 = s3.run_stage3(X_my, myopia_df, names, cfg,
                       train_idx=train_idx, test_idx=test_idx, command="all")
    r4 = s4.run_stage4(X_my, myopia_df, names, cfg,
                       train_idx=train_idx, test_idx=test_idx, command="all")

    summary = {
        "n_imagens_coorte": int(len(cohort)),
        "n_normal": int((cohort["label"] == 0).sum()),
        "n_miopia": int((cohort["label"] == 1).sum()),
        "n_features": len(names),
        "etapa2_melhor": r2["best_model"],
        "etapa2_metricas": r2["results"][r2["best_model"]],
        "etapa3_melhor": r3["best_model"],
        "etapa3_metricas": {k: v for k, v in r3["results"][r3["best_model"]].items()
                            if k != "report"},
        "etapa4_melhor": r4["best_model"],
        "etapa4_metricas": r4["results"][r4["best_model"]],
        "aviso": "Alvos das Etapas 3 e 4 sao SINTETICOS (proxy MDI), nao clinicos.",
    }
    save_json(summary, C.REPORTS_DIR / "resumo_pipeline.json")
    LOGGER.info("################### PIPELINE CONCLUIDO ###################")
    return summary