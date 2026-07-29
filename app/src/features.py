"""
features.py
===========
Extrai um VETOR DE FEATURES interpretaveis de cada imagem pre-processada.

Por que features "handcrafted" e nao CNN?
-----------------------------------------
O escopo do TCC fixa o scikit-learn como biblioteca de treino. Modelos classicos
(SVM, Random Forest, etc.) nao operam sobre pixels crus 224x224x3 = 150.528
dimensoes de forma eficiente -- isso levaria a maldicao da dimensionalidade. A
engenharia de atributos traduz o conhecimento clinico da miopia patologica em
numeros:

  * COR (RGB/HSV/LAB): a miopia degenerativa clareia o fundo (atrofia coroidal)
    e expoe vasos coroidais -> muda media/desvio/assimetria dos canais.
  * TEXTURA (LBP): o "tessellated fundus" e essencialmente uma assinatura de
    textura; Local Binary Patterns capturam micro-padroes de forma robusta a
    iluminacao.
  * GRADIENTE/BORDAS (Sobel): atrofia peripapilar e crescentes miopicos criam
    bordas de alto contraste.
  * PERFIL RADIAL: a tesselacao e mais intensa na periferia -> razao
    centro/periferia de brilho e de cor discrimina severidade.

Cada feature tem NOME (feature_names) para permitir analise de importancia.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import skew

from . import config as C
from .preprocessing import get_preprocessed
from .utils import LOGGER, parallel_map

_EPS = 1e-6

# -----------------------------------------------------------------------------
# Tabela LBP uniforme (precomputada uma vez)
# -----------------------------------------------------------------------------
def _build_uniform_lbp_table() -> tuple[np.ndarray, int]:
    """Mapa 256 -> indice de bin uniforme (58 uniformes + 1 nao-uniforme)."""
    def transitions(v: int) -> int:
        bits = [(v >> i) & 1 for i in range(8)]
        return sum(bits[i] != bits[(i + 1) % 8] for i in range(8))

    uniforms = [v for v in range(256) if transitions(v) <= 2]
    table = np.full(256, len(uniforms), dtype=np.int32)  # nao-uniforme -> ultimo bin
    for i, v in enumerate(uniforms):
        table[v] = i
    return table, len(uniforms) + 1


_LBP_TABLE, _LBP_BINS = _build_uniform_lbp_table()


def _lbp_hist(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Histograma normalizado de LBP uniforme (P=8, R=1)."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    nbrs = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:],
        g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2],
    ]
    code = np.zeros_like(c, dtype=np.uint8)
    for i, nb in enumerate(nbrs):
        code |= (nb >= c).astype(np.uint8) << i
    binned = _LBP_TABLE[code]
    m = mask[1:-1, 1:-1]
    vals = binned[m]
    if vals.size == 0:
        return np.zeros(_LBP_BINS, dtype=np.float32)
    hist = np.bincount(vals, minlength=_LBP_BINS).astype(np.float32)
    return hist / (hist.sum() + _EPS)


# -----------------------------------------------------------------------------
# Blocos de features
# -----------------------------------------------------------------------------
def _masked_stats(channel: np.ndarray, mask: np.ndarray) -> list[float]:
    vals = channel[mask].astype(np.float32)
    if vals.size == 0:
        return [0.0, 0.0, 0.0]
    std = float(vals.std())
    # Assimetria e indefinida quando o canal e (quase) constante; evita o
    # RuntimeWarning de "catastrophic cancellation" retornando 0.
    sk = float(skew(vals)) if std > 1e-3 else 0.0
    return [float(vals.mean()), std, sk]


def _color_features(img: np.ndarray, mask: np.ndarray) -> tuple[list[float], list[str]]:
    feats, names = [], []
    spaces = {
        "rgb": img,
        "hsv": cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
        "lab": cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
    }
    for sp, arr in spaces.items():
        for ci, cname in enumerate(sp):
            mean, std, sk = _masked_stats(arr[:, :, ci], mask)
            feats += [mean, std, sk]
            names += [f"{sp}_{cname}_mean", f"{sp}_{cname}_std", f"{sp}_{cname}_skew"]
    # Histogramas de cor (RGB 16 bins/canal + Hue 16 bins)
    for ci, cname in enumerate("rgb"):
        h = cv2.calcHist([img], [ci], mask.astype(np.uint8), [16], [0, 256]).flatten()
        h = h / (h.sum() + _EPS)
        feats += h.tolist()
        names += [f"hist_{cname}_{b}" for b in range(16)]
    hsv = spaces["hsv"]
    hh = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [16], [0, 180]).flatten()
    hh = hh / (hh.sum() + _EPS)
    feats += hh.tolist()
    names += [f"hist_hue_{b}" for b in range(16)]
    return feats, names


def _texture_features(gray: np.ndarray, mask: np.ndarray) -> tuple[list[float], list[str]]:
    lbp = _lbp_hist(gray, mask)
    names = [f"lbp_{i}" for i in range(_LBP_BINS)]
    # Nitidez/riqueza de textura via variancia do Laplaciano.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    feats = lbp.tolist() + [float(lap[mask].var()) if mask.any() else 0.0]
    names += ["laplacian_var"]
    return feats, names


def _gradient_features(gray: np.ndarray, mask: np.ndarray) -> tuple[list[float], list[str]]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) + np.pi)  # [0, 2pi)
    m = mag[mask]
    feats = [float(m.mean()), float(m.std()),
             float((m > 40).mean()) if m.size else 0.0]  # densidade de borda
    names = ["grad_mag_mean", "grad_mag_std", "edge_density"]
    hist, _ = np.histogram(ang[mask], bins=9, range=(0, 2 * np.pi),
                           weights=mag[mask] if mask.any() else None)
    hist = hist / (hist.sum() + _EPS)
    feats += hist.tolist()
    names += [f"grad_ori_{i}" for i in range(9)]
    return feats, names


def _radial_features(img: np.ndarray, gray: np.ndarray,
                     mask: np.ndarray) -> tuple[list[float], list[str]]:
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / (r.max() + _EPS)
    R, G = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32)
    rg_ratio = R / (G + _EPS)
    feats, names = [], []
    for lo, hi, tag in [(0.0, 0.33, "centro"), (0.33, 0.66, "meio"), (0.66, 1.01, "perif")]:
        ring = mask & (r >= lo) & (r < hi)
        if ring.any():
            feats += [float(gray[ring].mean()), float(rg_ratio[ring].mean())]
        else:
            feats += [0.0, 0.0]
        names += [f"radial_{tag}_brilho", f"radial_{tag}_rg"]
    return feats, names


# -----------------------------------------------------------------------------
# API publica
# -----------------------------------------------------------------------------
def extract_features(img_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extrai o vetor de features de UMA imagem pre-processada (RGB uint8)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > C.FUNDUS_CROP_THRESHOLD  # ignora padding preto

    feats: list[float] = []
    names: list[str] = []
    for block in (
        _color_features(img_rgb, mask),
        _texture_features(gray, mask),
        _gradient_features(gray, mask),
        _radial_features(img_rgb, gray, mask),
    ):
        f, n = block
        feats += f
        names += n
    return np.asarray(feats, dtype=np.float32), names


# Nomes das features (constante -- extraidos de uma imagem-dummy uma unica vez).
FEATURE_NAMES: list[str] = extract_features(np.full((C.IMAGE_SIZE, C.IMAGE_SIZE, 3),
                                                    128, dtype=np.uint8))[1]


def build_feature_matrix(cohort, cfg: C.RunConfig | None = None):
    """
    Constroi a matriz X (n_amostras x n_features) para a coorte, com cache em
    .npz alinhado por `filename` (garante que X e y permanecam consistentes).
    Retorna (X, feature_names).
    """
    cfg = cfg or C.RunConfig()
    filenames = cohort["filename"].tolist()

    if C.FEATURES_CACHE.exists() and not cfg.force_recompute:
        data = np.load(C.FEATURES_CACHE, allow_pickle=True)
        cached = list(data["filenames"])
        if cached == filenames:
            LOGGER.info("Matriz de features carregada do cache (%s).", C.FEATURES_CACHE)
            return data["X"], list(data["feature_names"])
        LOGGER.info("Cache de features desatualizado -> recomputando.")

    rows = list(cohort[["filename", "path"]].itertuples(index=False, name=None))

    def _work(item):
        filename, path = item
        img = get_preprocessed(filename, path, force=cfg.force_recompute)
        return extract_features(img)[0]

    LOGGER.info("Extraindo features de %d imagens...", len(rows))
    vecs = parallel_map(_work, rows, n_jobs=cfg.n_jobs, desc="Features")
    X = np.vstack(vecs).astype(np.float32)

    C.FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.FEATURES_CACHE, X=X,
                        filenames=np.array(filenames, dtype=object),
                        feature_names=np.array(FEATURE_NAMES, dtype=object))
    LOGGER.info("Features: X.shape=%s (%d atributos). Cache salvo.",
                X.shape, len(FEATURE_NAMES))
    return X, FEATURE_NAMES