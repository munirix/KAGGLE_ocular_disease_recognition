"""
stage3_severity.py -- ETAPA 3
=============================
Classificacao ORDINAL da severidade da miopia em 3 classes: leve < alta < magna.
Roda apenas sobre a coorte de miopia. Os rotulos sao o PROXY SINTETICO derivado
do MDI (ver labeling.py) -- as figuras carregam o aviso correspondente.

Notas de modelagem
------------------
* Coorte pequena (~260 imagens) e desbalanceada -> `class_weight="balanced"` e
  validacao cruzada estratificada com nº de folds adaptado a menor classe.
* Comparamos Regressao Logistica multinomial (baseline linear) e Random Forest.
* Metricas macro (tratam as 3 classes igualmente, ignorando a frequencia):
  balanced accuracy e F1-macro sao as mais honestas aqui.
* A severidade e ordinal; tratamo-la como multiclasse (simples e transparente) e
  discutimos na analise critica quando um modelo ordinal dedicado valeria a pena.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from . import config as C
from . import visualization as viz
from .utils import LOGGER, save_json, timer

_SEV_TO_INT = {s: i for i, s in enumerate(C.SEVERITY_CLASSES)}


def encode_severity(severity: np.ndarray) -> np.ndarray:
    """Converte rotulos textuais (leve/alta/magna) em inteiros ordinais 0/1/2."""
    return np.array([_SEV_TO_INT[s] for s in severity], dtype=int)


def _build_models(cfg: C.RunConfig) -> dict[str, Pipeline]:
    p = C.STAGE3
    cw = p.use_class_weight
    return {
        "logreg_multinomial": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight=cw,
                                       random_state=cfg.seed)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=p.rf_n_estimators, max_depth=p.rf_max_depth,
                min_samples_leaf=p.rf_min_samples_leaf, class_weight=cw,
                n_jobs=cfg.n_jobs, random_state=cfg.seed)),
        ]),
    }


def run_stage3(X: np.ndarray, severity: np.ndarray, feature_names: list[str],
               cfg: C.RunConfig | None = None) -> dict:
    """Treina/avalia a classificacao de severidade e gera as figuras."""
    cfg = cfg or C.RunConfig()
    y = encode_severity(severity)
    LOGGER.info("=== ETAPA 3: Severidade | X=%s | classes=%s ===",
                X.shape, np.bincount(y).tolist())

    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(C.CV_FOLDS, min_class))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=cfg.seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)

    results, fitted = {}, {}
    for name, model in _build_models(cfg).items():
        with timer(f"Etapa3:{name}"):
            cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv,
                                    scoring="f1_macro", n_jobs=cfg.n_jobs)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results[name] = {
                "cv_f1_macro_mean": float(cv_f1.mean()),
                "cv_f1_macro_std": float(cv_f1.std()),
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
                "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
                "report": classification_report(
                    y_te, y_pred, target_names=C.SEVERITY_CLASSES,
                    output_dict=True, zero_division=0),
            }
            fitted[name] = model
            LOGGER.info("%s -> f1_macro=%.3f bal_acc=%.3f",
                        name, results[name]["f1_macro"],
                        results[name]["balanced_accuracy"])

    best = max(results, key=lambda m: results[m]["f1_macro"])
    LOGGER.info("Melhor modelo Etapa 3: %s (F1-macro=%.3f)",
                best, results[best]["f1_macro"])

    y_pred_best = fitted[best].predict(X_te)
    viz.plot_confusion_matrix(y_te, y_pred_best, C.SEVERITY_CLASSES,
                              f"Etapa 3 - Severidade ({best})",
                              "stage3_confusion.png", synthetic=True)
    viz.plot_model_comparison(results, "f1_macro",
                              "Etapa 3 - F1-macro por modelo",
                              "stage3_f1_comparacao.png")
    if "random_forest" in fitted:
        rf = fitted["random_forest"].named_steps["clf"]
        viz.plot_feature_importance(feature_names, rf.feature_importances_, 20,
                                    "Etapa 3 - Top 20 features (Random Forest)",
                                    "stage3_feature_importance.png")

    joblib.dump(fitted[best], C.MODELS_DIR / "stage3_best_model.joblib")
    save_json({"best_model": best, "results": results},
              C.REPORTS_DIR / "stage3_metrics.json")
    return {"best_model": best, "results": results, "fitted": fitted}