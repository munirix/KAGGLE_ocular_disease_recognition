"""
stage2_classification.py -- ETAPA 2
===================================
Classificacao binaria POR IMAGEM: Normal (0) x Miopia (1).

Estrategia
----------
* Compara tres familias de algoritmos com vieses indutivos distintos:
  - Regressao Logistica (linear, interpretavel, baseline honesto);
  - SVM-RBF (fronteiras nao-lineares em espaco de features de dimensao media);
  - Random Forest (nao-linear, robusto a escala e a features irrelevantes,
    fornece importancia de atributos).
* Desbalanceamento ~11:1 tratado com `class_weight="balanced"` (reponderacao da
  funcao de perda) em vez de reamostragem -- mantemos a distribuicao real do
  teste e evitamos vazamento por oversampling antes do split.
* Todos os estimadores ficam dentro de um Pipeline com StandardScaler: o
  escalonamento e AJUSTADO apenas no treino (dentro da CV), evitando data leakage.

Duas melhorias metodologicas desta etapa
----------------------------------------
1. BUSCA DE HIPERPARAMETROS com HalvingGridSearchCV (successive halving): em vez
   de fixar os hiperparametros "no chute", buscamo-los maximizando a Average
   Precision na validacao cruzada do TREINO. O Halving e mais barato que o
   GridSearch exaustivo (comeca com poucas amostras/muitos candidatos e vai
   promovendo os melhores). Ativavel por `cfg.stage2_tune`.
2. LIMIAR DE DECISAO POR SENSIBILIDADE: o corte padrao de 0.5 e clinicamente
   arbitrario num rastreio. Escolhemos o limiar sobre escores OUT-OF-FOLD do
   treino (via cross_val_predict) -- portanto SEM vazamento do teste -- como o
   menor corte cuja sensibilidade (recall da miopia) atinge o alvo, e reportamos
   as metricas do teste tanto no ponto padrao quanto no ponto otimizado.

Metricas apropriadas para dados desbalanceados: balanced accuracy, F1/recall da
classe positiva, especificidade, ROC-AUC e, sobretudo, Average Precision (PR-AUC).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (HalvingGridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import joblib

from . import config as C
from . import visualization as viz
from .utils import LOGGER, save_json, timer


def _build_models(cfg: C.RunConfig) -> dict[str, Pipeline]:
    p = C.STAGE2
    cw = p.use_class_weight
    catalog = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=p.C_logreg, class_weight=cw,
                                       max_iter=5000, random_state=cfg.seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=p.svm_C, gamma=p.svm_gamma, kernel="rbf",
                        class_weight=cw, probability=True, random_state=cfg.seed)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=p.rf_n_estimators, max_depth=p.rf_max_depth,
                min_samples_leaf=p.rf_min_samples_leaf, class_weight=cw,
                n_jobs=cfg.n_jobs, random_state=cfg.seed)),
        ]),
    }
    return {k: catalog[k] for k in cfg.stage2_models if k in catalog}


def _positive_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Score continuo da classe positiva (proba se houver, senao decisao)."""
    try:
        return model.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        return model.decision_function(X)


def _oof_positive_scores(model: Pipeline, X: np.ndarray, y: np.ndarray,
                         cv, n_jobs: int) -> np.ndarray:
    """
    Escores out-of-fold da classe positiva no TREINO. Cada amostra e pontuada por
    um modelo que NAO a viu no ajuste -> base honesta (sem vazamento) para
    escolher o limiar. Usa proba quando disponivel; senao, a funcao de decisao.
    """
    try:
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba",
                                  n_jobs=n_jobs)
        return proba[:, 1]
    except (AttributeError, NotImplementedError, ValueError):
        return cross_val_predict(model, X, y, cv=cv, method="decision_function",
                                 n_jobs=n_jobs)


def _select_threshold(y_true: np.ndarray, y_score: np.ndarray,
                      target_sensitivity: float) -> tuple[float, str]:
    """
    Escolhe o limiar de decisao a partir de escores (idealmente out-of-fold).

    Regra: entre os limiares cuja sensibilidade (recall da miopia) >= alvo,
    escolhe o de MAIOR precisao (equivalente ao maior limiar que ainda satisfaz o
    alvo). Se o alvo for inatingivel, cai para o ponto de maximo J de Youden
    (tpr - fpr), que equilibra sensibilidade e especificidade.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # prec[i]/rec[i] correspondem a thr[i] (o ultimo par nao tem limiar).
    best_thr, best_prec = None, -1.0
    for i, t in enumerate(thr):
        if rec[i] >= target_sensitivity and prec[i] > best_prec:
            best_prec, best_thr = float(prec[i]), float(t)
    if best_thr is not None:
        return best_thr, f"sensibilidade>={target_sensitivity:.2f}"
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    j_best = int(np.argmax(tpr - fpr))
    return float(thr_roc[j_best]), "youden_j (alvo de sensibilidade inatingivel)"


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metricas no ponto de operacao (classe positiva = Miopia)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_myopia": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_myopia": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1_myopia": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_stage2(X: np.ndarray, y: np.ndarray, feature_names: list[str],
               cfg: C.RunConfig | None = None) -> dict:
    """Treina/avalia os modelos da Etapa 2 e gera todas as figuras."""
    cfg = cfg or C.RunConfig()
    LOGGER.info("=== ETAPA 2: Normal x Miopia | X=%s | tune=%s | alvo_sens=%.2f ===",
                X.shape, cfg.stage2_tune, cfg.stage2_target_sensitivity)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=cfg.seed)

    models = _build_models(cfg)
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=cfg.seed)

    results: dict[str, dict] = {}
    roc_curves: dict[str, dict] = {}
    pr_curves: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    scores_te: dict[str, np.ndarray] = {}

    for name, model in models.items():
        with timer(f"Etapa2:{name}"):
            # ---- 1) Ajuste: com busca de hiperparametros (Halving) ou fixo ----
            grid = C.STAGE2_PARAM_GRIDS.get(name)
            if cfg.stage2_tune and grid:
                # Piso de recurso limitado ao tamanho do treino (execucoes pequenas
                # viram uma unica rodada, i.e., busca em grade completa).
                min_res = min(C.STAGE2_HALVING_MIN_RESOURCES, len(X_tr))
                search = HalvingGridSearchCV(
                    model, grid, scoring="average_precision",
                    factor=C.STAGE2_HALVING_FACTOR, min_resources=min_res,
                    cv=cv, refit=True, random_state=cfg.seed, n_jobs=cfg.n_jobs)
                search.fit(X_tr, y_tr)
                estimator = search.best_estimator_        # reajustado em todo X_tr
                best_params = dict(search.best_params_)
            else:
                estimator = model.fit(X_tr, y_tr)
                best_params = {}

            # CV-AP HONESTA e comparavel: o best_score_ do Halving vem de um
            # SUBCONJUNTO (successive halving) e nao e comparavel entre modelos.
            # Reavaliamos o estimador escolhido em TODO o treino, com a mesma CV,
            # para selecionar o melhor modelo sem espiar o teste.
            cv_ap = float(cross_val_score(
                estimator, X_tr, y_tr, cv=cv,
                scoring="average_precision", n_jobs=cfg.n_jobs).mean())

            # ---- 2) Limiar via escores OUT-OF-FOLD do treino (sem vazamento) ----
            oof = _oof_positive_scores(estimator, X_tr, y_tr, cv, cfg.n_jobs)
            thr, thr_rule = _select_threshold(
                y_tr, oof, cfg.stage2_target_sensitivity)
            default_thr = 0.5 if hasattr(estimator, "predict_proba") else 0.0
            thresholds[name] = thr

            # ---- 3) Avaliacao no teste (curvas/AUC/AP: independentes do limiar) ----
            y_score = _positive_scores(estimator, X_te)
            scores_te[name] = y_score
            fpr, tpr, _ = roc_curve(y_te, y_score)
            prec, rec, _ = precision_recall_curve(y_te, y_score)
            auc = float(roc_auc_score(y_te, y_score))
            ap = float(average_precision_score(y_te, y_score))

            # ---- 4) Metricas nos dois pontos de operacao (padrao x otimizado) ----
            metrics_default = _binary_metrics(y_te, (y_score >= default_thr).astype(int))
            metrics_tuned = _binary_metrics(y_te, (y_score >= thr).astype(int))

            results[name] = {
                "best_params": best_params,
                "cv_average_precision": cv_ap,
                "roc_auc": auc,
                "average_precision": ap,
                "threshold_default": float(default_thr),
                "threshold_tuned": float(thr),
                "threshold_rule": thr_rule,
                "target_sensitivity": float(cfg.stage2_target_sensitivity),
                "default": metrics_default,
                "tuned": metrics_tuned,
            }
            roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            pr_curves[name] = {"precision": prec, "recall": rec, "ap": ap}
            fitted[name] = estimator
            LOGGER.info(
                "%s -> CV-AP=%.3f AP=%.3f AUC=%.3f | limiar %.3f (%s) | "
                "recall_M %.3f->%.3f prec_M %.3f->%.3f",
                name, cv_ap, ap, auc, thr, thr_rule,
                metrics_default["recall_myopia"], metrics_tuned["recall_myopia"],
                metrics_default["precision_myopia"], metrics_tuned["precision_myopia"])

    # Melhor modelo pela Average Precision de VALIDACAO CRUZADA (nao espia o teste).
    best = max(results, key=lambda m: results[m]["cv_average_precision"])
    LOGGER.info("Melhor modelo Etapa 2: %s (CV-AP=%.3f | AP_teste=%.3f)",
                best, results[best]["cv_average_precision"],
                results[best]["average_precision"])

    # ---- Figuras ----
    viz.plot_roc_curves(roc_curves, "stage2_roc.png")
    viz.plot_pr_curves(pr_curves, "stage2_pr.png")
    viz.plot_model_comparison(results, "average_precision",
                              "Etapa 2 - Average Precision por modelo",
                              "stage2_ap_comparacao.png")
    viz.plot_threshold_analysis(
        y_te, scores_te[best], thresholds[best], cfg.stage2_target_sensitivity,
        f"Etapa 2 - Escolha do limiar por sensibilidade ({best})",
        "stage2_threshold.png")
    # Matriz de confusao no ponto PADRAO (0,5) e no ponto OTIMIZADO por sensibilidade.
    viz.plot_confusion_matrix(
        y_te, (scores_te[best] >= results[best]["threshold_default"]).astype(int),
        C.BINARY_NAMES, f"Etapa 2 - Confusao no limiar padrao 0,5 ({best})",
        "stage2_confusion.png")
    viz.plot_confusion_matrix(
        y_te, (scores_te[best] >= thresholds[best]).astype(int),
        C.BINARY_NAMES,
        f"Etapa 2 - Confusao no limiar otimizado {thresholds[best]:.3f} ({best})",
        "stage2_confusion_tuned.png")
    if "random_forest" in fitted:
        rf = fitted["random_forest"].named_steps["clf"]
        viz.plot_feature_importance(feature_names, rf.feature_importances_, 20,
                                    "Etapa 2 - Top 20 features (Random Forest)",
                                    "stage2_feature_importance.png")

    # ---- Persistencia ----
    joblib.dump(fitted[best], C.MODELS_DIR / "stage2_best_model.joblib")
    operating_point = {
        "best_model": best,
        "threshold": float(thresholds[best]),
        "threshold_rule": results[best]["threshold_rule"],
        "target_sensitivity": float(cfg.stage2_target_sensitivity),
        "tuned": bool(cfg.stage2_tune),
    }
    save_json({"best_model": best, "operating_point": operating_point,
               "results": results},
              C.REPORTS_DIR / "stage2_metrics.json")
    return {"best_model": best, "results": results, "fitted": fitted,
            "thresholds": thresholds, "operating_point": operating_point,
            "split": (X_te, y_te)}