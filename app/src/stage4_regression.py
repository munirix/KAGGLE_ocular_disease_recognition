"""
stage4_regression.py -- ETAPA 4
===============================
Regressao do GRAU refrativo (equivalente esferico, em dioptrias) sobre a coorte
de miopia. O alvo `diopter` e SINTETICO (labeling.py); as figuras avisam isso.

Notas de modelagem
------------------
* Comparamos quatro regressores com vieses distintos:
  - Ridge (linear regularizado, baseline, controla colinearidade das features);
  - SVR-RBF (nao-linear, robusto a outliers via margem epsilon);
  - Random Forest Regressor (nao-linear, sem premissa de forma funcional);
  - Gradient Boosting (aditivo, costuma dar o menor erro em dados tabulares).
* Ridge e SVR entram em Pipeline com StandardScaler (sensiveis a escala);
  modelos de arvore dispensam escalonamento.
* Metricas: R² (variancia explicada), MAE (erro medio em dioptrias, unidade
  clinica interpretavel) e RMSE (penaliza erros grandes).
* Validacao cruzada (R² e MAE) no conjunto de treino para estimar generalizacao.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import joblib

from . import config as C
from . import visualization as viz
from .utils import LOGGER, save_json, timer


def _build_models(cfg: C.RunConfig) -> dict[str, Pipeline]:
    p = C.STAGE4
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=p.ridge_alpha, random_state=cfg.seed)),
        ]),
        "svr_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=p.svr_C, epsilon=p.svr_epsilon, kernel="rbf")),
        ]),
        "random_forest": Pipeline([
            ("reg", RandomForestRegressor(
                n_estimators=p.rf_n_estimators, min_samples_leaf=p.rf_min_samples_leaf,
                n_jobs=cfg.n_jobs, random_state=cfg.seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("reg", GradientBoostingRegressor(
                n_estimators=p.gbr_n_estimators, learning_rate=p.gbr_learning_rate,
                max_depth=p.gbr_max_depth, random_state=cfg.seed)),
        ]),
    }


def run_stage4(X: np.ndarray, diopter: np.ndarray, feature_names: list[str],
               cfg: C.RunConfig | None = None, stratify=None) -> dict:
    """Treina/avalia os regressores e gera dispersao + residuos."""
    cfg = cfg or C.RunConfig()
    y = np.asarray(diopter, dtype=float)
    LOGGER.info("=== ETAPA 4: Regressao do grau | X=%s | y in [%.1f, %.1f] D ===",
                X.shape, y.min(), y.max())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=cfg.seed, stratify=stratify)
    cv = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=cfg.seed)

    results, fitted, preds = {}, {}, {}
    for name, model in _build_models(cfg).items():
        with timer(f"Etapa4:{name}"):
            cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="r2",
                                    n_jobs=cfg.n_jobs)
            cv_mae = -cross_val_score(model, X_tr, y_tr, cv=cv,
                                      scoring="neg_mean_absolute_error",
                                      n_jobs=cfg.n_jobs)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results[name] = {
                "cv_r2_mean": float(cv_r2.mean()),
                "cv_mae_mean": float(cv_mae.mean()),
                "r2": float(r2_score(y_te, y_pred)),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "rmse": float(root_mean_squared_error(y_te, y_pred)),
            }
            fitted[name] = model
            preds[name] = y_pred
            LOGGER.info("%s -> R2=%.3f MAE=%.2f D RMSE=%.2f D",
                        name, results[name]["r2"], results[name]["mae"],
                        results[name]["rmse"])

    best = max(results, key=lambda m: results[m]["r2"])
    LOGGER.info("Melhor modelo Etapa 4: %s (R2=%.3f, MAE=%.2f D)",
                best, results[best]["r2"], results[best]["mae"])

    viz.plot_regression_scatter(y_te, preds[best], results[best]["r2"],
                                results[best]["mae"], "stage4_scatter.png")
    viz.plot_residuals(y_te, preds[best], "stage4_residuos.png")
    viz.plot_model_comparison(
        {k: {"r2": max(v["r2"], 0.0)} for k, v in results.items()}, "r2",
        "Etapa 4 - R² por modelo (truncado em 0)", "stage4_r2_comparacao.png")

    joblib.dump(fitted[best], C.MODELS_DIR / "stage4_best_model.joblib")
    save_json({"best_model": best, "results": results},
              C.REPORTS_DIR / "stage4_metrics.json")
    return {"best_model": best, "results": results, "fitted": fitted}