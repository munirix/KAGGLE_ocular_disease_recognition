"""
stage4_regression.py -- ETAPA 4 (DEMONSTRACAO)
==============================================
Regressao de um equivalente esferico SINTETICO (em dioptrias) sobre a coorte
de miopia. O alvo `diopter` e derivado heuristicamente das imagens (labeling.py).
Esta etapa e uma demonstracao arquitetural: nao estima refracao real nem
produz evidencia clinica.

O split e a validacao cruzada sao agrupados por paciente. A selecao do melhor
modelo usa R² medio de CV, nao o teste.

Os rotulos devem ser gerados antes (tipicamente por pipeline.prepare_myopia)
usando estatisticas apenas do treino; caso contrario, esta etapa os gera a
partir de todo o conjunto para permitir execucao autonoma (com aviso).
"""
from __future__ import annotations

import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C
from . import labeling, visualization as viz
from .logging_utils import LOGGER
from .provenance import build_manifest, file_sha256
from .utils import save_json, timer


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


def _split_grouped(X: np.ndarray, groups: np.ndarray,
                   cfg: C.RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Split agrupado por paciente usando GroupShuffleSplit."""
    n_unique = len(np.unique(groups))
    if n_unique < 2:
        raise ValueError("Nao e possivel dividir: menos de 2 grupos de pacientes.")
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=cfg.seed)
    return next(gss.split(X, groups=groups))


def _r2_scorer(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """R^2 silencioso para folds muito pequenos (evita UndefinedMetricWarning)."""
    y_pred = estimator.predict(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(r2_score(y, y_pred))


def run_stage4(X: np.ndarray, myopia_df: pd.DataFrame, feature_names: list[str],
               cfg: C.RunConfig | None = None,
               train_idx: np.ndarray | None = None,
               test_idx: np.ndarray | None = None,
               command: str = "stage4",
               persist: bool = True) -> dict:
    """Treina/avalia os regressores (demonstracao) e gera figuras."""
    cfg = cfg or C.RunConfig()
    groups = myopia_df[C.GROUP_COL].to_numpy()

    # Se os rotulos ainda nao existirem, divide primeiro por paciente e depois
    # gera os alvos apenas no treino, evitando vazamento mesmo em execucao autonoma.
    if "diopter" not in myopia_df.columns:
        if train_idx is None or test_idx is None:
            train_idx, test_idx = _split_grouped(X, groups, cfg)
        myopia_df = labeling.assign_synthetic_targets(myopia_df, cfg, fit_idx=train_idx)

    y = myopia_df["diopter"].to_numpy(dtype=float)
    filenames = myopia_df["filename"].to_numpy()
    patient_ids = myopia_df[C.GROUP_COL].to_numpy()
    LOGGER.info("=== ETAPA 4: Grau refrativo SINTETICO (DEMO) | X=%s | y in [%.1f, %.1f] D | pacientes=%d ===",
                X.shape, y.min(), y.max(), len(np.unique(groups)))

    # Split agrupado por paciente (~25% teste).
    if train_idx is None or test_idx is None:
        train_idx, test_idx = _split_grouped(X, groups, cfg)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = groups[train_idx]

    # CV agrupada para regressao: numero maximo de dobras, mas garantindo
    # pelo menos ~2 grupos por fold (evita R^2 indefinido em subamostras).
    n_unique_tr = len(np.unique(g_tr))
    cv_splits = max(2, min(C.CV_FOLDS, n_unique_tr // 2))
    cv = GroupKFold(n_splits=cv_splits)

    results, fitted, preds = {}, {}, {}
    for name, model in _build_models(cfg).items():
        with timer(f"Etapa4:{name}"):
            cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, groups=g_tr,
                                    scoring=_r2_scorer, n_jobs=cfg.n_jobs)
            cv_mae = -cross_val_score(model, X_tr, y_tr, cv=cv, groups=g_tr,
                                      scoring="neg_mean_absolute_error",
                                      n_jobs=cfg.n_jobs)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                test_r2 = float(r2_score(y_te, y_pred))
            results[name] = {
                "cv_r2_mean": float(cv_r2.mean()),
                "cv_mae_mean": float(cv_mae.mean()),
                "r2": test_r2,
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "rmse": float(root_mean_squared_error(y_te, y_pred)),
            }
            fitted[name] = model
            preds[name] = y_pred
            LOGGER.info("%s -> CV-R²=%.3f CV-MAE=%.2f D | test R²=%.3f MAE=%.2f D RMSE=%.2f D",
                        name, results[name]["cv_r2_mean"], results[name]["cv_mae_mean"],
                        results[name]["r2"], results[name]["mae"], results[name]["rmse"])

    best = max(results, key=lambda m: results[m]["cv_r2_mean"])
    LOGGER.info("Melhor modelo Etapa 4 (DEMO): %s (CV-R²=%.3f | test R²=%.3f, MAE=%.2f D)",
                best, results[best]["cv_r2_mean"], results[best]["r2"], results[best]["mae"])

    if persist:
        viz.plot_regression_scatter(y_te, preds[best], results[best]["r2"],
                                    results[best]["mae"], "stage4_scatter.png")
        viz.plot_residuals(y_te, preds[best], "stage4_residuos.png")
        viz.plot_model_comparison(
            {k: {"r2": max(v["cv_r2_mean"], 0.0)} for k, v in results.items()}, "r2",
            "Etapa 4 - R² de CV agrupada por modelo (truncado em 0)",
            "stage4_r2_comparacao.png", synthetic=True)

        joblib.dump(fitted[best], C.MODELS_DIR / "stage4_best_model.joblib")

        # Predicoes por imagem (OOF no treino, modelo final no teste).
        oof_pred = cross_val_predict(fitted[best], X_tr, y_tr, cv=cv, groups=g_tr,
                                     n_jobs=cfg.n_jobs)
        preds_rows = []
        for j, i in enumerate(train_idx):
            preds_rows.append({
                "filename": str(filenames[i]),
                "patient_id": str(patient_ids[i]),
                "partition": "train",
                "diopter_true": float(y[i]),
                "diopter_pred": float(oof_pred[j]),
                "model": best,
            })
        for j, i in enumerate(test_idx):
            preds_rows.append({
                "filename": str(filenames[i]),
                "patient_id": str(patient_ids[i]),
                "partition": "test",
                "diopter_true": float(y[i]),
                "diopter_pred": float(preds[best][j]),
                "model": best,
            })
        C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        preds_path = C.PREDICTIONS_DIR / "stage4_predictions.csv"
        pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
        LOGGER.info("Predicoes da Etapa 4 salvas em %s", preds_path)

    split_summary = {
        "n_train_images": int(len(train_idx)),
        "n_test_images": int(len(test_idx)),
        "n_train_patients": int(len(set(groups[train_idx]))),
        "n_test_patients": int(len(set(groups[test_idx]))),
        "n_overlap_patients": int(len(set(groups[train_idx]) & set(groups[test_idx]))),
    }

    if persist:
        payload = {
            "best_model": best,
            "best_criterion": "cv_r2_mean",
            "warning": "DEMONSTRACAO COM ALVO SINTETICO - SEM VALIDADE CLINICA",
            "results": results,
            "split_summary": split_summary,
        }
        save_json(payload, C.REPORTS_DIR / "stage4_metrics.json")

        manifest = build_manifest(cfg, command=command,
                                  extra={"stage": "stage4_demo",
                                         "data_file_hash": file_sha256(C.FULL_DF_CSV),
                                         "cohort_size": len(y),
                                         "split_summary": split_summary})
        save_json(manifest, C.REPORTS_DIR / "stage4_manifest.json")

        # Secao de resultados legivel (estilo stage2_results.txt)
        res = results[best]
        lines = [
            "=== Resultados da Etapa 4 (Grau refrativo sintetico) - DEMONSTRACAO ===",
            "Aviso: alvos sinteticos - sem validade clinica",
            f"Melhor modelo: {best}",
            f"Particao: {split_summary['n_train_images']} treino / {split_summary['n_test_images']} teste",
            f"Pacientes em comum: {split_summary['n_overlap_patients']}",
            f"CV R2: {res['cv_r2_mean']:.3f} | CV MAE: {res['cv_mae_mean']:.2f} D",
            "",
            "Metricas no teste:",
            f"  R2:   {res['r2']:.4f}",
            f"  MAE:  {res['mae']:.2f} D",
            f"  RMSE: {res['rmse']:.2f} D",
        ]
        results_path = C.REPORTS_DIR / "stage4_results.txt"
        results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info("Resultados da Etapa 4 salvos em %s", results_path)

    return {"best_model": best, "results": results, "fitted": fitted,
            "split_summary": split_summary}