"""
config.py
=========
Ponto unico de configuracao do pipeline de triagem do rotulo ocular de miopia.

Centralizar caminhos, sementes e hiperparametros em um so lugar e uma decisao
arquitetural deliberada: garante REPRODUTIBILIDADE (requisito inegociavel em um
TCC, pois a banca deve conseguir reexecutar e obter os mesmos numeros) e evita
"magic numbers" espalhados pelo codigo.

Alvo de linguagem: Python 3.13.x (testado em Python 3.13.4).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. CAMINHOS
# -----------------------------------------------------------------------------
# Resolvidos a partir da localizacao deste arquivo (app/src/config.py) para que
# o pipeline funcione independentemente do diretorio de trabalho do usuario.
SRC_DIR: Path = Path(__file__).resolve().parent
APP_DIR: Path = SRC_DIR.parent
DATA_DIR: Path = APP_DIR / "data"
OUTPUTS_DIR: Path = APP_DIR / "outputs"

# Fontes de dados brutas (fornecidas pelo ODIR-5K do Kaggle).
FULL_DF_CSV: Path = DATA_DIR / "full_df.csv"
DATA_XLSX: Path = DATA_DIR / "ODIR-5K" / "ODIR-5K" / "data.xlsx"
RAW_TRAIN_DIR: Path = DATA_DIR / "ODIR-5K" / "ODIR-5K" / "Training Images"
RAW_TEST_DIR: Path = DATA_DIR / "ODIR-5K" / "ODIR-5K" / "Testing Images"
# Imagens ja pre-processadas que acompanham o dataset (usadas apenas como
# fallback / comparacao; o pipeline gera o SEU proprio pre-processamento).
PROVIDED_PREPROCESSED_DIR: Path = DATA_DIR / "preprocessed_images"

# Artefatos de saida.
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
CACHE_DIR: Path = OUTPUTS_DIR / "cache"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# -----------------------------------------------------------------------------
# 2. REPRODUTIBILIDADE
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20        # holdout de teste (realizado via GroupKFold 5 folds)
CV_FOLDS: int = 5              # validacao cruzada estratificada agrupada por paciente
GROUP_COL: str = "patient_id"  # coluna que identifica o paciente (dois olhos juntos)

# -----------------------------------------------------------------------------
# 3. PARAMETROS DE PRE-PROCESSAMENTO (Etapa 1)
# -----------------------------------------------------------------------------
IMAGE_SIZE: int = 224          # lado do quadrado final (px)
# CLAHE = Contrast Limited Adaptive Histogram Equalization.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
FUNDUS_CROP_THRESHOLD: int = 7 # limiar (0-255) para recortar a borda preta

# -----------------------------------------------------------------------------
# 3.1 PARAMETROS DE EXTRACAO DE FEATURES (afetam o cache features_<hash>.npz)
# -----------------------------------------------------------------------------
COLOR_HIST_BINS: int = 16      # bins por canal RGB
HUE_HIST_BINS: int = 16        # bins do histograma de matiz HSV
LBP_P: int = 8                 # vizinhos do LBP (deve ser <= 16 para a tabela uniforme)
LBP_R: int = 1                 # raio do LBP
GRADIENT_BINS: int = 9         # bins de orientacao do gradiente Sobel
N_RADIAL_RINGS: int = 3        # aneis radiais (centro, meio, periferia)
# Versao interna do extrator de features; incremente quando alterar a logica de
# _lbp_hist, _color_features etc. (nao basta mudar os parametros numericos).
FEATURES_VERSION: str = "3"

# -----------------------------------------------------------------------------
# 3.2 HASHES DE CONFIGURACAO (invalidam caches automaticamente)
# -----------------------------------------------------------------------------
# Quando constantes de pre-processamento/extracao mudam, os caches antigos
# devem ser automaticamente ignorados (a nao ser que --force seja passado).

def _preproc_config_hash() -> str:
    """Hash curto dos parametros de pre-processamento que alteram pixels."""
    payload = {
        "IMAGE_SIZE": IMAGE_SIZE,
        "CLAHE_CLIP_LIMIT": float(CLAHE_CLIP_LIMIT),
        "CLAHE_TILE_GRID": list(CLAHE_TILE_GRID),
        "FUNDUS_CROP_THRESHOLD": FUNDUS_CROP_THRESHOLD,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def _features_config_hash() -> str:
    """Hash curto dos parametros que afetam o vetor de features."""
    payload = {
        "preproc": {
            "IMAGE_SIZE": IMAGE_SIZE,
            "CLAHE_CLIP_LIMIT": float(CLAHE_CLIP_LIMIT),
            "CLAHE_TILE_GRID": list(CLAHE_TILE_GRID),
            "FUNDUS_CROP_THRESHOLD": FUNDUS_CROP_THRESHOLD,
        },
        "features": {
            "COLOR_HIST_BINS": COLOR_HIST_BINS,
            "HUE_HIST_BINS": HUE_HIST_BINS,
            "LBP_P": LBP_P,
            "LBP_R": LBP_R,
            "GRADIENT_BINS": GRADIENT_BINS,
            "N_RADIAL_RINGS": N_RADIAL_RINGS,
            "FUNDUS_CROP_THRESHOLD": FUNDUS_CROP_THRESHOLD,
            "FEATURES_VERSION": FEATURES_VERSION,
        },
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


PREPROCESSED_CACHE_DIR: Path = CACHE_DIR / f"preprocessed_{_preproc_config_hash()}"
FEATURES_CACHE: Path = CACHE_DIR / f"features_{_features_config_hash()}.npz"


def ensure_dirs() -> None:
    """Cria toda a arvore de saida caso ainda nao exista."""
    for d in (
        OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, CACHE_DIR,
        REPORTS_DIR, PREPROCESSED_CACHE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 4. DEFINICAO DAS CLASSES / ROTULOS
# -----------------------------------------------------------------------------
# Etapa 2 - classificacao binaria por IMAGEM, com split agrupado por PACIENTE.
CLASS_NORMAL: str = "N"
CLASS_MYOPIA: str = "M"
BINARY_LABELS: dict[str, int] = {CLASS_NORMAL: 0, CLASS_MYOPIA: 1}
BINARY_NAMES: list[str] = ["Normal", "Miopia"]

# Palavras-chave (em minusculas) que definem cada coorte por imagem, a partir
# da coluna de diagnostic keywords do olho correspondente ao arquivo.
NORMAL_KEYWORD: str = "normal fundus"
MYOPIA_KEYWORDS: tuple[str, ...] = (
    "pathological myopia", "myopia retinopathy", "myopic retinopathy",
    "myopic maculopathy", "tessellated fundus", "myopic",
)

# Etapa 3 - severidade (ordinal). Ordem crescente de gravidade.
SEVERITY_CLASSES: list[str] = ["leve", "moderado", "alta", "magna"]

# -----------------------------------------------------------------------------
# 5. ALVOS SINTETICOS (Etapas 3 e 4) -- AVISO METODOLOGICO
# -----------------------------------------------------------------------------
# O ODIR-5K NAO contem grau de severidade nem equivalente esferico (dioptrias).
# ~90% das imagens de miopia sao rotuladas apenas como "pathological myopia".
# Portanto, os alvos de severidade e de grau numerico sao um PROXY SIMULADO,
# derivado de um "Indice de Degeneracao Miopica" (MDI) calculado a partir de
# features reais da imagem + ruido gaussiano controlado e reprodutivel.
# ESTES VALORES NAO SAO MEDIDAS CLINICAS. Servem exclusivamente para demonstrar
# a arquitetura das Etapas 3 e 4. Toda figura/relatorio derivado carrega o
# aviso "[ALVO SINTETICO]".

# Faixas de equivalente esferico (dioptrias, negativas -> miopia) por severidade.
# Baseadas em convencoes clinicas: alta miopia <= -6 D; miopia degenerativa
# (magna) tipicamente <= -10/-12 D.
DIOPTER_RANGES: dict[str, tuple[float, float]] = {
    "leve": (-0.1, -3.0),
    "moderado": (-3.0, -6.0),
    "alta": (-6.0, -12.0),
    "magna": (-12.0, -22.0),
}

# Quantis do MDI usados para discretizar as classes de SEVERITY_CLASSES.
# Devem ter comprimento len(SEVERITY_CLASSES) - 1 e estar em ordem crescente.
SEVERITY_QUANTILES: tuple[float, float, float] = (0.30, 0.55, 0.80)
DIOPTER_NOISE_STD: float = 1.2  # desvio-padrao do ruido (D) no alvo de regressao

# -----------------------------------------------------------------------------
# 6. HIPERPARAMETROS DOS MODELOS
# -----------------------------------------------------------------------------
N_JOBS: int = -1  # usa todos os nucleos disponiveis


@dataclass(frozen=True)
class Stage2Params:
    """Classificacao binaria Normal x Miopia."""
    c_logreg: float = 1.0
    svm_C: float = 10.0
    svm_gamma: str = "scale"
    rf_n_estimators: int = 400
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 2
    use_class_weight: str = "balanced"  # trata o desbalanceamento ~11:1


@dataclass(frozen=True)
class Stage3Params:
    """Classificacao de severidade (4 classes ordinais: leve/moderado/alta/magna)."""
    ordinal_alpha: float = 1.0  # regularizacao L2 do LogisticAT (mord)
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 1
    use_class_weight: str = "balanced"


@dataclass(frozen=True)
class Stage4Params:
    """Regressao do grau (equivalente esferico)."""
    ridge_alpha: float = 10.0
    svr_C: float = 10.0
    svr_epsilon: float = 0.5
    rf_n_estimators: int = 400
    rf_min_samples_leaf: int = 2
    gbr_n_estimators: int = 300
    gbr_learning_rate: float = 0.05
    gbr_max_depth: int = 3


STAGE2 = Stage2Params()
STAGE3 = Stage3Params()
STAGE4 = Stage4Params()

# -----------------------------------------------------------------------------
# 6.1 BUSCA DE HIPERPARAMETROS E LIMIAR DE DECISAO (Etapa 2)
# -----------------------------------------------------------------------------
# Em vez de "chutar" os hiperparametros de Stage2Params, a Etapa 2 pode busca-los
# com HalvingGridSearchCV (successive halving): mais barato que o GridSearchCV
# exaustivo -- comeca com poucas amostras e muitos candidatos e vai eliminando os
# piores. A metrica de selecao e a Average Precision (foco na classe rara), medida
# APENAS na validacao cruzada do conjunto de treino (o teste nunca e tocado).
STAGE2_TUNE: bool = True
STAGE2_HALVING_FACTOR: int = 3  # taxa de eliminacao/promocao entre "rungs"
# Recurso (nº de amostras) do PRIMEIRO rung. Com a classe positiva rara (~8%),
# rungs iniciais pequenos dao estimativas de AP ruidosas e podem descartar boas
# configuracoes. Este piso garante ~50 positivos ja na primeira rodada; e limitado
# a n_amostras de treino em execucoes pequenas (vira uma busca em grade unica).
STAGE2_HALVING_MIN_RESOURCES: int = 600

# Grids enxutos de proposito: o Halving ja e eficiente e a classe positiva
# (miopia) e rara -- grids grandes so aumentariam a variancia da estimativa.
# As chaves usam o prefixo do passo do Pipeline ("clf__<param>").
STAGE2_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "dummy_stratified": {},
    "dummy_most_frequent": {},
    "logreg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "svm_rbf": {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", 1e-3, 1e-2, 1e-1],
    },
    "random_forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
    },
}

# Otimizacao do LIMIAR de decisao por SENSIBILIDADE. Em rastreio de patologia, o
# corte padrao de 0,5 e clinicamente arbitrario: priorizamos nao deixar passar
# casos de miopia (recall/sensibilidade alto). O limiar e escolhido sobre escores
# OUT-OF-FOLD do treino (sem vazamento) como o menor corte cuja sensibilidade
# atinge o alvo abaixo, entre eles o de maior precisao.
STAGE2_TARGET_SENSITIVITY: float = 0.90


@dataclass
class RunConfig:
    """Configuracao de uma execucao concreta (permite overrides por CLI)."""
    max_per_class: int | None = None  # limita amostras/classe (smoke test)
    force_recompute: bool = False
    n_jobs: int = N_JOBS
    seed: int = RANDOM_SEED
    log_file: str | None = None  # caminho do log de execucao (execution_<timestamp>.log)
    stage2_models: list[str] = field(
        default_factory=lambda: ["dummy_stratified", "dummy_most_frequent",
                                "logreg", "svm_rbf", "random_forest"]
    )
    stage2_tune: bool = STAGE2_TUNE
    stage2_target_sensitivity: float = STAGE2_TARGET_SENSITIVITY