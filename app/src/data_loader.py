"""
data_loader.py
==============
Responsavel por transformar os metadados brutos do ODIR-5K (full_df.csv) na
COORTE por imagem usada em todo o pipeline.

Decisao de modelagem central
----------------------------
O ODIR-5K e anotado por PACIENTE (colunas N, D, G, C, A, H, M, O refletem os
dois olhos juntos). Modelar "por paciente" mistura dois olhos que podem ter
diagnosticos diferentes. Optamos por classificar POR IMAGEM (um olho por
amostra), pois:
  1. e a unidade natural de uma fotografia de fundo de olho;
  2. dobra o numero de amostras;
  3. remove o rotulo ambiguo de pacientes com olhos discordantes.

Para isso, o rotulo de cada linha vem da diagnostic keyword do OLHO indicado
pela coluna `filename` (ex.: `0_right.jpg` -> `Right-Diagnostic Keywords`).
  - Normal : keyword == "normal fundus".
  - Miopia : keyword contem algum termo de miopia (config.MYOPIA_KEYWORDS).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C
from .utils import LOGGER

# -----------------------------------------------------------------------------
# Metadados brutos
# -----------------------------------------------------------------------------
def load_full_df() -> pd.DataFrame:
    """Carrega o full_df.csv como veio do Kaggle (uma linha por imagem)."""
    if not C.FULL_DF_CSV.exists():
        raise FileNotFoundError(f"Nao encontrei {C.FULL_DF_CSV}")
    df = pd.read_csv(C.FULL_DF_CSV)
    LOGGER.info("full_df.csv carregado: %d linhas x %d colunas", *df.shape)
    return df


def _eye_of(filename: str) -> str:
    fn = str(filename).lower()
    if fn.endswith("_right.jpg"):
        return "right"
    if fn.endswith("_left.jpg"):
        return "left"
    return "unknown"


def _eye_keyword(row: pd.Series) -> str:
    """Keyword diagnostica do olho correspondente ao arquivo da linha."""
    eye = _eye_of(row["filename"])
    col = "Right-Diagnostic Keywords" if eye == "right" else "Left-Diagnostic Keywords"
    return str(row.get(col, "")).strip().lower()


def resolve_image_path(filename: str) -> Path | None:
    """
    Localiza o arquivo de imagem tentando, em ordem: Training, Testing e o
    diretorio de imagens pre-processadas fornecido. Retorna None se nao achar.
    """
    for base in (C.RAW_TRAIN_DIR, C.RAW_TEST_DIR, C.PROVIDED_PREPROCESSED_DIR):
        p = base / filename
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Coorte por imagem (Normal x Miopia)
# -----------------------------------------------------------------------------
def _classify_row(eye_kw: str) -> str | None:
    """Mapeia a keyword do olho para 'N', 'M' ou None (descartada)."""
    if eye_kw == C.NORMAL_KEYWORD:
        return C.CLASS_NORMAL
    if any(term in eye_kw for term in C.MYOPIA_KEYWORDS):
        return C.CLASS_MYOPIA
    return None


def build_cohort(cfg: C.RunConfig | None = None) -> pd.DataFrame:
    """
    Constroi o DataFrame da coorte por imagem, com colunas:
      image_id, filename, eye, path, eye_keyword, class_str (N/M), label (0/1),
      age, sex.

    `cfg.max_per_class` permite subamostrar de forma estratificada e reprodutivel
    (util para smoke tests). Linhas sem arquivo em disco sao descartadas.
    """
    cfg = cfg or C.RunConfig()
    df = load_full_df()

    df = df.copy()
    df["eye"] = df["filename"].map(_eye_of)
    df["eye_keyword"] = df.apply(_eye_keyword, axis=1)
    df["class_str"] = df["eye_keyword"].map(_classify_row)

    cohort = df[df["class_str"].notna()].copy()
    cohort["label"] = cohort["class_str"].map(C.BINARY_LABELS).astype(int)

    # Resolve caminhos e descarta o que nao existe em disco.
    cohort["path"] = cohort["filename"].map(
        lambda f: str(p) if (p := resolve_image_path(f)) else None
    )
    missing = cohort["path"].isna().sum()
    if missing:
        LOGGER.warning("%d imagens da coorte nao foram encontradas em disco.", missing)
    cohort = cohort[cohort["path"].notna()].copy()

    # Subamostragem estratificada opcional (loop explicito evita o
    # comportamento depreciado de groupby.apply sobre a coluna de agrupamento).
    if cfg.max_per_class is not None:
        parts = [
            g.sample(min(len(g), cfg.max_per_class), random_state=cfg.seed)
            for _, g in cohort.groupby("class_str")
        ]
        cohort = pd.concat(parts).reset_index(drop=True)

    cohort = cohort.rename(columns={"ID": "image_id",
                                    "Patient Age": "age",
                                    "Patient Sex": "sex"})
    keep = ["image_id", "filename", "eye", "path", "eye_keyword",
            "class_str", "label", "age", "sex"]
    cohort = cohort[keep].reset_index(drop=True)

    n_norm = int((cohort["label"] == 0).sum())
    n_myop = int((cohort["label"] == 1).sum())
    ratio = n_norm / max(n_myop, 1)
    LOGGER.info("Coorte final: %d imagens | Normal=%d Miopia=%d (razao %.1f:1)",
                len(cohort), n_norm, n_myop, ratio)

    return cohort


def myopia_subset(cohort: pd.DataFrame) -> pd.DataFrame:
    """Recorte apenas com imagens de miopia (base das Etapas 3 e 4)."""
    return cohort[cohort["label"] == 1].reset_index(drop=True)