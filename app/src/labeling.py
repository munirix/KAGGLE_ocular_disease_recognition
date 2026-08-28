"""
labeling.py -- Alvos sinteticos das Etapas 3 (severidade) e 4 (grau/dioptrias)
=============================================================================
O ODIR-5K nao disponibiliza graus de severidade nem dioptrias explicitas para
a coorte de miopia. Para viabilizar a demonstracao metodologica do pipeline
multietapa (classificacao binaria -> ordinal -> regressao), derivamos um Indice
de Deformacao Miopica (MDI in [0, 1]) a partir de uma combinacao linear de 5 medidas
interpretaveis da propria imagem, clinicamente motivadas:

  * pallor           - clareamento do fundo (atrofia coroidal deixa o fundo palido)
  * low_saturation   - perda de saturacao de cor (idem)
  * periphery_edge   - densidade de bordas na periferia (tesselado coroidal)
  * bright_frac      - fracao de pixels muito claros (crescentes/atrofia peripapilar)
  * rg_periphery     - razao R/G na periferia (exposicao da coroide)

O MDI (in [0,1]) e discretizado em severidade (leve/moderado/alta/magna) por quantis e
mapeado para dioptrias por interpolacao linear por partes + ruido gaussiano
semeado.

>>> ESTES ALVOS SAO SINTETICOS. NAO SAO MEDIDAS CLINICAS. <<<
Toda figura/relatorio derivado exibe o aviso "[ALVO SINTETICO]". Em producao,
estes rotulos deveriam vir de graduacao META-PM por oftalmologista e de
refracao medida (equivalente esferico real).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .preprocessing import get_preprocessed
from .logging_utils import LOGGER
from .utils import parallel_map

_EPS = 1e-6

SYNTHETIC_DISCLAIMER = "[ALVO SINTETICO - derivado do MDI, nao e medida clinica]"

# Pesos documentados de cada componente no MDI (somam 1.0).
MDI_WEIGHTS: dict[str, float] = {
    "pallor": 0.30,
    "low_saturation": 0.20,
    "periphery_edge": 0.25,
    "bright_frac": 0.15,
    "rg_periphery": 0.10,
}


def compute_mdi_components(img: np.ndarray) -> dict[str, float]:
    """Mede os 5 componentes interpretaveis do MDI em uma imagem pre-processada."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > C.FUNDUS_CROP_THRESHOLD
    if not mask.any():
        return {k: 0.0 for k in MDI_WEIGHTS}

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0

    # Perfil radial para isolar a periferia.
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    r /= (r.max() + _EPS)
    periph = mask & (r >= 0.66)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    rg = R / (G + _EPS)

    return {
        "pallor": float(gray[mask].mean() / 255.0),
        "low_saturation": float(1.0 - sat[mask].mean()),
        "periphery_edge": float((mag[periph] > 40).mean()) if periph.any() else 0.0,
        "bright_frac": float((gray[mask] > 200).mean()),
        "rg_periphery": float(rg[periph].mean()) if periph.any() else 0.0,
    }


MDI_COMPONENT_COLS: list[str] = list(MDI_WEIGHTS.keys())


def _compute_mdi_components_for_df(df: pd.DataFrame,
                                  cfg: C.RunConfig) -> pd.DataFrame:
    """Calcula os 5 componentes do MDI para cada imagem e anexa ao DataFrame."""
    df = df.copy().reset_index(drop=True)
    rows = list(df[["filename", "path"]].itertuples(index=False, name=None))

    def _work(item):
        filename, path = item
        img = get_preprocessed(filename, path, force=cfg.force_recompute)
        return compute_mdi_components(img)

    LOGGER.info("Calculando componentes do MDI de %d imagens de miopia...", len(rows))
    comps = parallel_map(_work, rows, n_jobs=cfg.n_jobs, desc="MDI")
    comp_df = pd.DataFrame(comps)
    for col in MDI_COMPONENT_COLS:
        df[col] = comp_df[col].astype(float)
    return df


def assign_synthetic_targets(myopia_df: pd.DataFrame,
                             cfg: C.RunConfig | None = None,
                             fit_idx: np.ndarray | None = None) -> pd.DataFrame:
    """Acrescenta a `myopia_df` as colunas: mdi, severity (leve/moderado/alta/magna)
    e diopter (equivalente esferico sintetico).

    Se `fit_idx` for fornecido, a normalizacao, os quantis e as ancora de
    dioptria sao estimados APENAS nesse subconjunto (tipicamente o treino),
    evitando que estatisticas do teste vazem para os rotulos. Se `fit_idx`
    for None, os rotulos sao gerados a partir de toda a coorte (modo legacy
    / demonstracao sem divisao previa).
    """
    cfg = cfg or C.RunConfig()
    df = myopia_df.copy().reset_index(drop=True)

    if not all(col in df.columns for col in MDI_COMPONENT_COLS):
        df = _compute_mdi_components_for_df(df, cfg)

    comps = df[MDI_COMPONENT_COLS].to_numpy(dtype=float)
    fit = np.arange(len(df)) if fit_idx is None else np.asarray(fit_idx)
    fit_comps = comps[fit]

    # Estatisticas de normalizacao vindas apenas do conjunto de ajuste.
    means = fit_comps.mean(axis=0)
    stds = fit_comps.std(axis=0) + _EPS

    # MDI = soma ponderada dos componentes padronizados.
    z = (comps - means) / stds
    weights = np.array([MDI_WEIGHTS[c] for c in MDI_COMPONENT_COLS], dtype=float)
    mdi_raw = (z * weights).sum(axis=1)

    # Normaliza o MDI usando apenas os extremos do ajuste.
    lo = float(mdi_raw[fit].min())
    hi = float(mdi_raw[fit].max())
    mdi = (mdi_raw - lo) / (hi - lo + _EPS)
    df["mdi"] = mdi

    # Severidade por quantis estimados no ajuste.
    n_classes = len(C.SEVERITY_CLASSES)
    if len(C.SEVERITY_QUANTILES) != n_classes - 1:
        raise ValueError(
            f"SEVERITY_QUANTILES deve ter {n_classes - 1} elementos; "
            f"tem {len(C.SEVERITY_QUANTILES)}"
        )

    qs = np.asarray(np.quantile(mdi[fit], C.SEVERITY_QUANTILES), dtype=float).ravel()
    bins = np.digitize(mdi, qs)
    df["severity"] = np.array([C.SEVERITY_CLASSES[i] for i in bins], dtype=object)

    # Dioptrias: interpolacao linear por partes. Os nos (xp) sao os quantis do MDI
    # e os extremos do ajuste; os valores (fp) sao os limites superiores de cada
    # classe em DIOPTER_RANGES, seguidos do limite inferior da classe mais grave.
    # Ordem das tuplas em DIOPTER_RANGES nao importa (sao ordenadas aqui).
    diopter_bounds = [sorted(C.DIOPTER_RANGES[cls]) for cls in C.SEVERITY_CLASSES]
    lo = float(mdi[fit].min())
    hi = float(mdi[fit].max())
    # Garante nos distintos para a interpolacao.
    qs = np.maximum(qs, np.nextafter(lo, hi))
    qs = np.minimum(qs, np.nextafter(hi, lo))
    xp = np.asarray([lo, *qs.tolist(), hi], dtype=float)
    fp = np.asarray([bounds[1] for bounds in diopter_bounds] + [diopter_bounds[-1][0]], dtype=float)
    diopter = np.interp(mdi, xp, fp)
    rng = np.random.default_rng(cfg.seed)
    diopter = diopter + rng.normal(0.0, C.DIOPTER_NOISE_STD, size=len(diopter))
    clip_lo = min(bounds[0] for bounds in diopter_bounds)
    clip_hi = max(bounds[1] for bounds in diopter_bounds)
    df["diopter"] = np.clip(diopter, clip_lo, clip_hi)

    dist = df["severity"].value_counts().reindex(C.SEVERITY_CLASSES).fillna(0).astype(int)
    if fit_idx is None:
        LOGGER.warning("ALVOS SINTETICOS gerados (nao clinicos) usando TODA a coorte. "
                       "Estatisticas do conjunto de teste podem ter vazado para os rotulos.")
    else:
        LOGGER.info("ALVOS SINTETICOS gerados (nao clinicos) a partir de %d amostras de ajuste "
                    "e aplicados a %d amostras.", len(fit), len(df))
    LOGGER.info("Severidade: %s | Dioptria media=%.1f D", dist.to_dict(), df["diopter"].mean())
    return df