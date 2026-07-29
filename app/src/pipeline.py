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

from . import config as C
from . import data_loader, features, labeling, preprocessing
from . import stage2_classification as s2
from . import stage3_severity as s3
from . import stage4_regression as s4
from . import visualization as viz
from .utils import LOGGER, save_json, set_global_seed, timer

# -----------------------------------------------------------------------------
# Preparacao compartilhada (coorte + matriz de features), com cache
# -----------------------------------------------------------------------------
def prepare(cfg: C.RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    C.ensure_dirs()
    set_global_seed(cfg.seed)
    full_df = data_loader.load_full_df()
    cohort = data_loader.build_cohort(cfg)
    X, names = features.build_feature_matrix(cohort, cfg)
    return full_df, cohort, X, names


def prepare_myopia(cfg: C.RunConfig, cohort: pd.DataFrame,
                   X: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Recorta a coorte de miopia (alinhando X) e gera os alvos sinteticos."""
    mask = (cohort["label"] == 1).to_numpy()
    X_my = X[mask]
    myopia_df = data_loader.myopia_subset(cohort)           # mesma ordem de `mask`
    myopia_df = labeling.assign_synthetic_targets(myopia_df, cfg)
    assert len(myopia_df) == X_my.shape[0], "Desalinhamento X_my x myopia_df"
    return X_my, myopia_df


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
        _, myopia_df = prepare_myopia(cfg, cohort, X)
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
    for _, row in samples.iterrows():
        steps = preprocessing.preprocess_image(row["path"], return_steps=True)
        label = "Miopia" if row["class_str"] == "M" else "Normal"
        viz.plot_preprocessing_steps(
            steps, f"Etapa 1 - {label} ({row['filename']})",
            f"stage1_steps_{row['class_str']}_{row['filename'].replace('.jpg','')}.png")
        pairs.append((steps["1_original"], steps["4_clahe"], label))
    viz.plot_original_vs_preprocessed(pairs)


# -----------------------------------------------------------------------------
# Etapas 2, 3 e 4
# -----------------------------------------------------------------------------
def run_stage2(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    _, cohort, X, names = prepare(cfg)
    return s2.run_stage2(X, cohort["label"].to_numpy(), names, cfg)


def run_stage3(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    _, cohort, X, names = prepare(cfg)
    X_my, myopia_df = prepare_myopia(cfg, cohort, X)
    return s3.run_stage3(X_my, myopia_df["severity"].to_numpy(), names, cfg)


def run_stage4(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    _, cohort, X, names = prepare(cfg)
    X_my, myopia_df = prepare_myopia(cfg, cohort, X)
    return s4.run_stage4(X_my, myopia_df["diopter"].to_numpy(), names, cfg,
                         stratify=myopia_df["severity"].to_numpy())


# -----------------------------------------------------------------------------
# Pipeline completo
# -----------------------------------------------------------------------------
def run_all(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    LOGGER.info("############### PIPELINE COMPLETO ###############")
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
            f"stage1_steps_{row['class_str']}_{row['filename'].replace('.jpg','')}.png")
        pairs.append((steps["1_original"], steps["4_clahe"], label))
    viz.plot_original_vs_preprocessed(pairs)

    # EDA.
    viz.plot_label_distribution_full(full_df)
    viz.plot_binary_distribution(cohort)
    viz.plot_demographics(cohort)

    # Etapa 2.
    r2 = s2.run_stage2(X, cohort["label"].to_numpy(), names, cfg)

    # Etapas 3 e 4 (coorte de miopia + alvos sinteticos).
    X_my, myopia_df = prepare_myopia(cfg, cohort, X)
    viz.plot_severity_distribution(myopia_df)
    viz.plot_diopter_distribution(myopia_df)
    r3 = s3.run_stage3(X_my, myopia_df["severity"].to_numpy(), names, cfg)
    r4 = s4.run_stage4(X_my, myopia_df["diopter"].to_numpy(), names, cfg,
                       stratify=myopia_df["severity"].to_numpy())

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
    LOGGER.info("############### PIPELINE CONCLUIDO ###############")
    return summary