"""
labeling.py -- ALVOS DAS ETAPAS 3 E 4  (AVISO METODOLOGICO IMPORTANTE)
=============================================================================
O ODIR-5K **nao possui** grau de severidade nem equivalente esferico
(dioptrias). Verificamos que ~98% das imagens de miopia sao rotuladas apenas
como "pathological myopia" -- ou seja, a abordagem por palavra-chave produz uma
unica classe e e inviavel para um problema de 3 classes.

Para ainda assim demonstrar a ARQUITETURA das Etapas 3 (severidade) e 4
(regressao do grau), derivamos os alvos de um PROXY TRANSPARENTE E REPRODUTIVEL:
o "Indice de Degeneracao Miopica" (MDI), calculado a partir de medidas
interpretaveis da propria imagem, clinicamente motivadas:

  * pallor         - clareamento do fundo (atrofia coroidal deixa o fundo palido)
  * low_saturation - perda de saturacao de cor (idem)
  * periphery_edge - densidade de bordas na periferia (tesselado coroidal)
  * bright_frac    - fracao de pixels muito claros (crescentes/atrofia peripapilar)
  * rg_periphery   - razao R/G na periferia (exposicao da coroide)

O MDI (E[0,1]) e discretizado em severidade (leve/alta/magna) por quantis e
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
from .utils import LOGGER, parallel_map

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
# Ancoras do mapeamento MDI -> equivalente esferico (dioptrias, negativas).
_DIOPTER_ANCHORS = [-3.0, -6.0, -12.0, -22.0]


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


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + _EPS)


def assign_synthetic_targets(myopia_df: pd.DataFrame,
                             cfg: C.RunConfig | None = None) -> pd.DataFrame:
    """
    Acrescenta a `myopia_df` as colunas: mdi, severity (leve/alta/magna) e
    diopter (equivalente esferico sintetico). Processo 100% reprodutivel.
    """
    cfg = cfg or C.RunConfig()
    df = myopia_df.copy().reset_index(drop=True)
    rows = list(df[["filename", "path"]].itertuples(index=False, name=None))

    def _work(item):
        filename, path = item
        img = get_preprocessed(filename, path, force=cfg.force_recompute)
        return compute_mdi_components(img)

    LOGGER.info("Calculando MDI de %d imagens de miopia...", len(rows))
    comps = parallel_map(_work, rows, n_jobs=cfg.n_jobs, desc="MDI")
    comp_df = pd.DataFrame(comps)

    # MDI = soma ponderada dos componentes padronizados (z-score na coorte).
    mdi_raw = np.zeros(len(comp_df), dtype=np.float64)
    for name, weight in MDI_WEIGHTS.items():
        mdi_raw += weight * _zscore(comp_df[name].to_numpy())
    mdi = (mdi_raw - mdi_raw.min()) / (mdi_raw.max() - mdi_raw.min() + _EPS)
    df["mdi"] = mdi

    # Severidade por quantis (controla o balanceamento da coorte).
    q1, q2 = np.quantile(mdi, C.SEVERITY_QUANTILES)
    sev = np.where(mdi < q1, "leve", np.where(mdi < q2, "alta", "magna"))
    df["severity"] = sev

    # Dioptrias: interpolacao linear por partes (ancorada nos quantis) + ruido.
    xp = [mdi.min(), q1, q2, mdi.max()]
    diopter = np.interp(mdi, xp, _DIOPTER_ANCHORS)
    rng = np.random.default_rng(cfg.seed)
    diopter = diopter + rng.normal(0.0, C.DIOPTER_NOISE_STD, size=len(diopter))
    df["diopter"] = np.clip(diopter, -24.0, -1.5)

    dist = df["severity"].value_counts().reindex(C.SEVERITY_CLASSES).fillna(0).astype(int)
    LOGGER.warning("ALVOS SINTETICOS gerados (nao clinicos). Severidade: %s | "
                   "Dioptria media=%.1f D", dist.to_dict(), df["diopter"].mean())

    return df