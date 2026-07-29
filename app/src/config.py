"""
config.py
=========
Ponto unico de configuracao do pipeline de diagnostico de miopia.

Centralizar caminhos, sementes e hiperparametros em um so lugar e uma decisao
arquitetural deliberada: garante REPRODUTIBILIDADE (requisito inegociavel em um
TCC, pois a banca deve conseguir reexecutar e obter os mesmos numeros) e evita
"magic numbers" espalhados pelo codigo.

Alvo de linguagem: Python 3.14.0 (o codigo tambem roda em 3.13.x).
"""
from __future__ import annotations

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
CACHE_DIR: Path = OUTPUTS_DIR / "cache"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
# Cache do pre-processamento (Etapa 1) e da matriz de features.
PREPROCESSED_CACHE_DIR: Path = CACHE_DIR / "preprocessed"
FEATURES_CACHE: Path = CACHE_DIR / "features.npz"
DATASET_CACHE: Path = CACHE_DIR / "dataset.parquet"


def ensure_dirs() -> None:
    """Cria toda a arvore de saida caso ainda nao exista."""
    for d in (
        OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, CACHE_DIR,
        REPORTS_DIR, PREPROCESSED_CACHE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. REPRODUTIBILIDADE
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20        # holdout de teste
CV_FOLDS: int = 5              # validacao cruzada estratificada

# -----------------------------------------------------------------------------
# 3. PARAMETROS DE PRE-PROCESSAMENTO (Etapa 1)
# -----------------------------------------------------------------------------
IMAGE_SIZE: int = 224          # lado do quadrado final (px)
# CLAHE = Contrast Limited Adaptive Histogram Equalization.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
FUNDUS_CROP_THRESHOLD: int = 7 # limiar (0-255) para recortar a borda preta

# -----------------------------------------------------------------------------
# 4. DEFINICAO DAS CLASSES / ROTULOS
# -----------------------------------------------------------------------------
# Etapa 2 - classificacao binaria por IMAGEM (nao por paciente).
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
SEVERITY_CLASSES: list[str] = ["leve", "alta", "magna"]


# -----------------------------------------------------------------------------
# 5. ALVOS SINTETICOS (Etapas 3 e 4) -- AVISO METODOLOGICO
# -----------------------------------------------------------------------------
# O ODIR-5K NAO contem grau de severidade nem equivalente esferico (dioptrias).
# ~98% das imagens de miopia sao rotuladas apenas como "pathological myopia".
# Portanto, os alvos de severidade e de grau numerico sao um PROXY SIMULADO,
# derivado de um "Indice de Degeneracao Miopica" (MDI) calculado a partir de
# features reais da imagem + ruido gaussiano controlado e reprodutivel.
# ESTES VALORES NAO SAO MEDIDAS CLINICAS. Servem exclusivamente para demonstrar
# a arquitetura das Etapas 3 e 4. Toda figura/relatorio derivado carrega o
# aviso "[ALVO SINTETICO]".
SYNTHETIC_TARGETS: bool = True

# Faixas de equivalente esferico (dioptrias, negativas -> miopia) por severidade.
# Baseadas em convencoes clinicas: alta miopia <= -6 D; miopia degenerativa
# (magna) tipicamente <= -10/-12 D.
DIOPTER_RANGES: dict[str, tuple[float, float]] = {
    "leve": (-3.0, -6.0),
    "alta": (-6.0, -12.0),
    "magna": (-12.0, -22.0),
}
# Quantis do MDI usados para discretizar em leve/alta/magna. Refletem uma
# prevalencia plausivel dentro de uma coorte ja diagnosticada como patologica.
SEVERITY_QUANTILES: tuple[float, float] = (0.45, 0.80) # < .45 leve; .45-.80 alta; > .80 magna
DIOPTER_NOISE_STD: float = 1.2  # desvio-padrao do ruido (D) no alvo de regressao

# -----------------------------------------------------------------------------
# 6. HIPERPARAMETROS DOS MODELOS
# -----------------------------------------------------------------------------
N_JOBS: int = -1  # usa todos os nucleos disponiveis

@dataclass(frozen=True)
class Stage2Params:
    """Classificacao binaria Normal x Miopia."""
    C_logreg: float = 1.0
    svm_C: float = 10.0
    svm_gamma: str = "scale"
    rf_n_estimators: int = 400
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 2
    use_class_weight: str = "balanced"  # trata o desbalanceamento ~11:1

@dataclass(frozen=True)
class Stage3Params:
    """Classificacao de severidade (3 classes)."""
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
# com HalvingGridSearchCV (successive halving): mais barato que o GridSearch
# exaustivo -- comeca com poucas amostras e muitos candidatos e vai eliminando os
# piores. A metrica de selecao e a Average Precision (foco na classe rara), medida
# APENAS na validacao cruzada do conjunto de treino (o teste nunca e tocado).
STAGE2_TUNE: bool = True
STAGE2_HALVING_FACTOR: int = 3   # taxa de eliminacao/promocao entre "rungs"
# Recurso (nº de amostras) do PRIMEIRO rung. Com a classe positiva rara (~8%),
# rungs iniciais pequenos dao estimativas de AP ruidosas e podem descartar boas
# configuracoes. Este piso garante ~50 positivos ja na primeira rodada; e limitado
# a n_amostras de treino em execucoes pequenas (vira uma busca em grade unica).
STAGE2_HALVING_MIN_RESOURCES: int = 600

# Grids enxutos de proposito: o Halving ja e eficiente e a classe positiva
# (miopia) e rara -- grids grandes so aumentariam a variancia da estimativa.
# As chaves usam o prefixo do passo do Pipeline ("clf__<param>").
STAGE2_PARAM_GRIDS: dict[str, dict[str, list]] = {
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
# corte padrao de 0.5 e clinicamente arbitrario: priorizamos nao deixar passar
# casos de miopia (recall/sensibilidade alto). O limiar e escolhido sobre escores
# OUT-OF-FOLD do treino (sem vazamento) como o menor corte cuja sensibilidade
# atinge o alvo abaixo, entre eles o de maior precisao.
STAGE2_TARGET_SENSITIVITY: float = 0.90

@dataclass
class RunConfig:
    """Configuracao de uma execucao concreta (permite overrides por CLI)."""
    max_per_class: int | None = None  # limita amostras/classe (smoke test)
    use_provided_preprocessed: bool = False
    force_recompute: bool = False
    n_jobs: int = N_JOBS
    seed: int = RANDOM_SEED
    stage2_models: list[str] = field(
        default_factory=lambda: ["logreg", "svm_rbf", "random_forest"]
    )
    stage2_tune: bool = STAGE2_TUNE
    stage2_target_sensitivity: float = STAGE2_TARGET_SENSITIVITY