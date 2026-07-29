"""
preprocessing.py -- ETAPA 1 do pipeline
=============================================================================
Transforma imagens de fundo de olho de RESOLUCOES VARIADAS (Canon, Zeiss, Kowa)
em tensores homogeneos, com contraste realcado, prontos para extracao de
features.

Sequencia de operacoes e a razao de cada uma:
  1. Recorte do disco de fundo (crop): as fotos vem com grande borda preta ao
     redor do circulo da retina. Recortar remove pixels irrelevantes, padroniza
     o enquadramento e reduz o vies de "quantidade de preto" entre cameras.
  2. Padding para quadrado: iguala o menor lado ao maior ANTES do resize, para
     nao distorcer a proporcao (uma retina esticada mudaria a textura, que e
     justamente o sinal clinico da miopia).
  3. Resize para 224x224: homogeneiza a resolucao (requisito central, dado o mix
     de cameras) e fixa a dimensionalidade das features.
  4. CLAHE (Contrast Limited Adaptive Histogram Equalization) no canal L do
     espaco LAB: realca contraste LOCAL (tesselado coroidal, atrofia
     peripapilar) sem estourar a cor global -- superior a equalizacao global.
  5. Normalizacao: versao float32 em [0,1] para etapas numericas.

Todas as etapas intermediarias podem ser retornadas (return_steps=True) para a
visualizacao "antes x depois" exigida no TCC.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .utils import LOGGER, parallel_map

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def read_rgb(path: str | Path) -> np.ndarray:
    """Le uma imagem do disco em RGB uint8 (OpenCV le como BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Falha ao ler a imagem: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Operacoes atomicas
# -----------------------------------------------------------------------------
def crop_fundus(img: np.ndarray, threshold: int = C.FUNDUS_CROP_THRESHOLD) -> np.ndarray:
    """Recorta a borda preta ao redor do circulo do fundo de olho."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Preenche com preto o lado menor para tornar a imagem quadrada."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    bottom, right = side - h - top, side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def resize_square(img: np.ndarray, size: int = C.IMAGE_SIZE) -> np.ndarray:
    """Redimensiona para size x size (INTER_AREA reduz sem serrilhado)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray,
                clip: float = C.CLAHE_CLIP_LIMIT,
                grid: tuple[int, int] = C.CLAHE_TILE_GRID) -> np.ndarray:
    """CLAHE aplicado ao canal de luminancia (L) do espaco LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)


def normalize01(img: np.ndarray) -> np.ndarray:
    """Converte uint8 [0,255] -> float32 [0,1]."""
    return img.astype(np.float32) / 255.0


# -----------------------------------------------------------------------------
# Pipeline completo
# -----------------------------------------------------------------------------
def preprocess_image(path: str | Path,
                     size: int = C.IMAGE_SIZE,
                     return_steps: bool = False):
    """
    Aplica Etapa 1 completa. Retorna a imagem final uint8 RGB (crop+square+
    resize+CLAHE). Se return_steps=True, retorna um dict com cada estagio,
    usado na visualizacao comparativa.
    """
    original = read_rgb(path)
    cropped = crop_fundus(original)
    squared = pad_to_square(cropped)
    resized = resize_square(squared, size)
    enhanced = apply_clahe(resized)

    if not return_steps:
        return enhanced
    return {
        "1_original": original,
        "2_cropped": cropped,
        "3_resized": resized,
        "4_clahe": enhanced,
        "5_normalized": normalize01(enhanced),
    }


# -----------------------------------------------------------------------------
# Cache em disco (evita reprocessar milhares de imagens a cada execucao)
# -----------------------------------------------------------------------------
def _cache_path(filename: str) -> Path:
    return C.PREPROCESSED_CACHE_DIR / f"{Path(filename).stem}.png"


def get_preprocessed(filename: str, src_path: str, force: bool = False) -> np.ndarray:
    """Retorna a imagem pre-processada, usando cache em PNG quando possivel."""
    cache = _cache_path(filename)
    if cache.exists() and not force:
        img = cv2.imread(str(cache), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = preprocess_image(src_path)
    C.PREPROCESSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out


def preprocess_cohort(cohort, cfg: C.RunConfig | None = None) -> None:
    """Pre-processa (e cacheia) todas as imagens da coorte em paralelo."""
    cfg = cfg or C.RunConfig()
    C.PREPROCESSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(cohort[["filename", "path"]].itertuples(index=False, name=None))

    def _work(item):
        filename, path = item
        get_preprocessed(filename, path, force=cfg.force_recompute)
        return True

    LOGGER.info("Pre-processando %d imagens (Etapa 1)...", len(rows))
    parallel_map(_work, rows, n_jobs=cfg.n_jobs, desc="Etapa 1 (pre-proc)")
    LOGGER.info("Pre-processamento concluido. Cache em %s", C.PREPROCESSED_CACHE_DIR)