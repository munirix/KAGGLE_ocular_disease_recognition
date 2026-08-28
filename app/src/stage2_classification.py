"""
stage2_classification.py -- ETAPA 2
===================================
Classificacao binaria POR IMAGEM (Normal=0 x Miopia=1) com split
AGRUPADO POR PACIENTE (dois olhos do mesmo paciente permanecem juntos).

Objetivos metodologicos
-----------------------
* Evitar vazamento de identidade/paciente entre treino e teste, usando
  StratifiedGroupKFold por `patient_id`.
* Incluir baselines ingenuos (DummyClassifier) para contextualizar as metricas.
* Selecionar o melhor modelo pela Average Precision de VALIDACAO CRUZADA
  agrupada, sem espiar o teste.
* Escolher o limiar de decisao a partir de escores OUT-OF-FOLD do treino,
  buscando sensibilidade-alvo.
* Persistir predicoes, indices do split e manifesto de reprodutibilidade.

A unidade de predicao continua sendo a imagem (um olho), mas a unidade de
independencia estatistica e o paciente. Todas as validacoes cruzadas respeitam
esse agrupamento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     HalvingGridSearchCV, StratifiedGroupKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import visualization as viz
from .logging_utils import LOGGER
from .provenance import build_manifest, file_sha256
from .utils import save_json, timer


def _min_groups_per_class(y: np.ndarray, groups: np.ndarray) -> int:
    """Numero minimo de grupos distintos presentes em cada classe."""
    per_class: dict[int, set] = {}
    for yi, gi in zip(y, groups):
        per_class.setdefault(int(yi), set()).add(gi)
    if not per_class:
        return 0
    return min(len(s) for s in per_class.values())


def _make_group_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   cfg: C.RunConfig) -> StratifiedGroupKFold | GroupKFold:
    """Constroi um validador cruzado agrupado. Preferencialmente usa
    StratifiedGroupKFold com um numero de dobras que nao force uma unica
    amostra/classe por fold (evita avisos e folds degenerados em subamostras).
    Se nao houver grupos suficientes em alguma classe, recorre a GroupKFold.
    O numero maximo de dobras e C.CV_FOLDS.
    """
    min_groups = _min_groups_per_class(y, groups)
    if min_groups >= 2:
        # Garante, no minimo, ~2 grupos da classe minoritaria por dobra.
        n_splits = min(C.CV_FOLDS, max(2, min_groups // 2))
        if n_splits >= 2:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    n_unique = len(np.unique(groups))
    if n_unique < 2:
        raise ValueError("Pelo menos 2 grupos de pacientes sao necessarios para CV.")
    n_splits = max(2, min(C.CV_FOLDS, n_unique))
    LOGGER.warning("Grupos insuficientes por classe para estratificacao; usando GroupKFold(%d).",
                   n_splits)
    return GroupKFold(n_splits=n_splits)


def _group_train_test_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            cfg: C.RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Split treino/teste agrupado por paciente. Se houver grupos suficientes,
    usa StratifiedGroupKFold com ~C.TEST_SIZE de teste (1/C.CV_FOLDS); senao,
    recorre a GroupShuffleSplit com C.TEST_SIZE para nao exaurir as classes
    em subamostras (smoke tests).
    """
    min_groups = _min_groups_per_class(y, groups)
    if min_groups >= 2 * C.CV_FOLDS:
        gss = StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=cfg.seed)
        return next(gss.split(X, y, groups))

    n_unique = len(np.unique(groups))
    if n_unique < 2:
        raise ValueError("Pelo menos 2 grupos de pacientes sao necessarios para o split.")
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=cfg.seed)
    return next(gss.split(X, groups=groups))


def _build_models(cfg: C.RunConfig) -> dict[str, Pipeline]:
    p = C.STAGE2
    cw = p.use_class_weight
    catalog = {
        "dummy_stratified": Pipeline([
            ("clf", DummyClassifier(strategy="stratified",
                                    random_state=cfg.seed)),
        ]),
        "dummy_most_frequent": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent",
                                    random_state=cfg.seed)),
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=p.c_logreg, class_weight=cw,
                                       max_iter=5000, random_state=cfg.seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=p.svm_C, gamma=p.svm_gamma, kernel="rbf",
                        class_weight=cw, probability=True,
                        random_state=cfg.seed)),
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
                         cv, n_jobs: int, groups: np.ndarray) -> np.ndarray:
    """Escores out-of-fold da classe positiva no TREINO, respeitando grupos.
    Cada amostra e pontuada por um modelo que NAO a viu no ajuste.
    """
    try:
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba",
                                  groups=groups, n_jobs=n_jobs)
        return proba[:, 1]
    except (AttributeError, NotImplementedError, ValueError):
        return cross_val_predict(model, X, y, cv=cv, method="decision_function",
                                 groups=groups, n_jobs=n_jobs)


def _select_threshold(y_true: np.ndarray, y_score: np.ndarray,
                      target_sensitivity: float) -> tuple[float, str]:
    """Escolhe o limiar de decisao a partir de escores (idealmente out-of-fold).

    Regra: entre os limiares cuja sensibilidade (recall da miopia) >= alvo,
    escolhe o de MAIOR precisao. Se o alvo for inatingivel, maximiza o indice
    de Youden (tpr - fpr).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    best_thr, best_prec = None, -1.0
    for i, t in enumerate(thr):
        if rec[i] >= target_sensitivity and prec[i] > best_prec:
            best_prec, best_thr = float(prec[i]), float(t)
    if best_thr is not None:
        return best_thr, f"sensibilidade>={target_sensitivity:.2f}"
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    j_best = int(np.argmax(tpr - fpr))
    return float(thr_roc[j_best]), "youden_j (alvo de sensibilidade inatingivel)"


def _wilson_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianca de Wilson para uma proporcao amostral."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if alpha == 0.05 else 1.96
    p = x / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _format_ci(ci: tuple[float, float]) -> str:
    """Formata intervalo de confianca como [a; b] com 3 casas decimais."""
    return f"[{ci[0]:.3f}; {ci[1]:.3f}]"


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metricas no ponto de operacao, com IC 95% de Wilson para proporcoes."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_myopia": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_myopia": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1_myopia": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "recall_myopia_ci95": _wilson_interval(tp, tp + fn),
        "specificity_ci95": _wilson_interval(tn, tn + fp),
        "precision_myopia_ci95": _wilson_interval(tp, tp + fp),
    }


def run_stage2(X: np.ndarray, cohort: pd.DataFrame,
               feature_names: list[str], cfg: C.RunConfig | None = None,
               command: str = "stage2") -> dict:
    """Treina/avalia os modelos da Etapa 2 e gera artefatos."""
    cfg = cfg or C.RunConfig()

    y = cohort["label"].to_numpy()
    groups = cohort[C.GROUP_COL].to_numpy()
    filenames = cohort["filename"].to_numpy()
    patient_ids = cohort[C.GROUP_COL].to_numpy()

    LOGGER.info("=== ETAPA 2: Normal x Miopia | X=%s | n=%d | pacientes=%d | "
                "tune=%s | alvo_sens=%.2f ===",
                X.shape, len(y), len(np.unique(groups)), cfg.stage2_tune,
                cfg.stage2_target_sensitivity)

    # -------------------------------------------------------------------------
    # Split treino/teste AGRUPADO por paciente.
    # O numero de dobras e ajustado ao numero de grupos por classe para evitar
    # erros com subamostras (smoke tests) e manter, no maximo, ~20% de teste.
    # -------------------------------------------------------------------------
    train_idx, test_idx = _group_train_test_split(X, y, groups, cfg)

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    groups_tr = groups[train_idx]

    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        LOGGER.warning("Vazamento de paciente detectado: %d pacientes no treino e teste",
                       len(overlap))
    else:
        LOGGER.info("Split limpo: 0 pacientes em comum entre treino e teste")

    # CV interna tambem agrupada por paciente (n dobras adaptativo).
    cv = _make_group_cv(X_tr, y_tr, groups_tr, cfg)

    models = _build_models(cfg)
    results: dict[str, dict] = {}
    roc_curves: dict[str, dict] = {}
    pr_curves: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    scores_te: dict[str, np.ndarray] = {}
    oof_scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        with timer(f"Etapa2:{name}"):
            # ---- 1) Ajuste (com busca de hiperparametros, quando houver grid) ----
            grid = C.STAGE2_PARAM_GRIDS.get(name)
            if cfg.stage2_tune and grid:
                min_res = min(C.STAGE2_HALVING_MIN_RESOURCES, len(X_tr))
                search = HalvingGridSearchCV(
                    model, grid, scoring="average_precision",
                    factor=C.STAGE2_HALVING_FACTOR, min_resources=min_res,
                    cv=cv, refit=True, random_state=cfg.seed,
                    n_jobs=cfg.n_jobs)
                search.fit(X_tr, y_tr, groups=groups_tr)
                estimator = search.best_estimator_
                best_params = dict(search.best_params_)
            else:
                estimator = model.fit(X_tr, y_tr)
                best_params = {}

            # ---- 2) CV-AP agrupada para selecao do melhor modelo ----
            cv_ap = float(cross_val_score(
                estimator, X_tr, y_tr, cv=cv,
                scoring="average_precision", groups=groups_tr,
                n_jobs=cfg.n_jobs).mean())

            # ---- 3) Limiar via escores OUT-OF-FOLD do treino ----
            if name.startswith("dummy"):
                # Baselines ingenuos nao sao calibrados por sensibilidade.
                default_thr = 0.5 if hasattr(estimator, "predict_proba") else 0.0
                thr = default_thr
                thr_rule = "baseline_dummy"
                oof = _oof_positive_scores(estimator, X_tr, y_tr, cv, cfg.n_jobs,
                                           groups=groups_tr)
            else:
                oof = _oof_positive_scores(estimator, X_tr, y_tr, cv, cfg.n_jobs,
                                           groups=groups_tr)
                try:
                    thr, thr_rule = _select_threshold(
                        y_tr, oof, cfg.stage2_target_sensitivity)
                except Exception:
                    thr = 0.5 if hasattr(estimator, "predict_proba") else 0.0
                    thr_rule = "fallback_default"
                default_thr = 0.5 if hasattr(estimator, "predict_proba") else 0.0
            oof_scores[name] = oof
            thresholds[name] = thr

            # ---- 4) Avaliacao no teste (curvas/AUC/AP independentes do limiar) ----
            y_score = _positive_scores(estimator, X_te)
            scores_te[name] = y_score
            fpr, tpr, _ = roc_curve(y_te, y_score)
            prec, rec, _ = precision_recall_curve(y_te, y_score)
            auc = float(roc_auc_score(y_te, y_score))
            ap = float(average_precision_score(y_te, y_score))

            # ---- 5) Metricas nos dois pontos de operacao ----
            metrics_default = _binary_metrics(
                y_te, (y_score >= default_thr).astype(int))
            metrics_tuned = _binary_metrics(
                y_te, (y_score >= thr).astype(int))

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

    # Melhor modelo pela CV-AP agrupada (nao espia o teste).
    best = max(results, key=lambda m: results[m]["cv_average_precision"])
    LOGGER.info("Melhor modelo Etapa 2: %s (CV-AP=%.3f | AP_teste=%.3f)",
                best, results[best]["cv_average_precision"],
                results[best]["average_precision"])

    # -------------------------------------------------------------------------
    # Figuras
    # -------------------------------------------------------------------------
    viz.plot_roc_curves(roc_curves, "stage2_roc.png")
    viz.plot_pr_curves(pr_curves, "stage2_pr.png")
    viz.plot_model_comparison(results, "average_precision",
                              "Etapa 2 - Average Precision por modelo",
                              "stage2_ap_comparacao.png")
    viz.plot_threshold_analysis(
        y_te, scores_te[best], thresholds[best], cfg.stage2_target_sensitivity,
        f"Etapa 2 - Escolha do limiar por sensibilidade ({best})",
        "stage2_threshold.png")
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

    # -------------------------------------------------------------------------
    # Persistencia: modelo, predicoes, split e manifesto
    # -------------------------------------------------------------------------
    C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = C.PREDICTIONS_DIR / "stage2_predictions.csv"
    rows = []
    for idx, part, scores in (
        (train_idx, "train", oof_scores[best]),
        (test_idx, "test", scores_te[best]),
    ):
        for j, i in enumerate(idx):
            score = float(scores[j])
            rows.append({
                "filename": str(filenames[i]),
                "patient_id": str(patient_ids[i]),
                "partition": part,
                "y_true": int(y[i]),
                "y_score": score,
                "y_pred_default": int(score >= results[best]["threshold_default"]),
                "y_pred_tuned": int(score >= thresholds[best]),
                "model": best,
            })
    pd.DataFrame(rows).to_csv(preds_path, index=False)
    LOGGER.info("Predicoes salvas em %s", preds_path)

    split_info = {
        "n_train_images": int(len(train_idx)),
        "n_test_images": int(len(test_idx)),
        "n_train_patients": int(len(set(groups[train_idx]))),
        "n_test_patients": int(len(set(groups[test_idx]))),
        "n_overlap_patients": int(len(overlap)),
        "train_idx": [int(i) for i in train_idx],
        "test_idx": [int(i) for i in test_idx],
        "train_files": [str(filenames[i]) for i in train_idx],
        "test_files": [str(filenames[i]) for i in test_idx],
    }
    save_json(split_info, C.REPORTS_DIR / "stage2_split.json")

    joblib.dump(fitted[best], C.MODELS_DIR / "stage2_best_model.joblib")
    operating_point = {
        "best_model": best,
        "threshold": float(thresholds[best]),
        "threshold_rule": results[best]["threshold_rule"],
        "target_sensitivity": float(cfg.stage2_target_sensitivity),
        "tuned": bool(cfg.stage2_tune),
    }
    metrics_payload = {
        "best_model": best,
        "operating_point": operating_point,
        "results": results,
        "split_summary": {k: v for k, v in split_info.items()
                          if k not in ("train_idx", "test_idx", "train_files", "test_files")},
    }
    save_json(metrics_payload, C.REPORTS_DIR / "stage2_metrics.json")

    manifest = build_manifest(
        cfg, command=command,
        extra={
            "stage": "stage2",
            "data_file_hash": file_sha256(C.FULL_DF_CSV),
            "cohort_size": int(len(cohort)),
            "n_features": len(feature_names),
            "best_model": best,
            "operating_point": operating_point,
            "split_summary": metrics_payload["split_summary"],
        })
    save_json(manifest, C.REPORTS_DIR / "stage2_manifest.json")

    # Secao de resultados legivel, incluindo intervalos de Wilson
    results_path = C.REPORTS_DIR / "stage2_results.txt"
    tuned_metrics = results[best]["tuned"]
    lines = [
        "=== Resultados da Etapa 2 (Normal x Miopia) ===",
        f"Melhor modelo: {best}",
        f"Particao: {split_info['n_train_images']} treino / {split_info['n_test_images']} teste",
        f"Pacientes em comum: {split_info['n_overlap_patients']}",
        f"Limiar de decisao: {thresholds[best]:.4f} ({results[best]['threshold_rule']})",
        f"AUC ROC no teste: {results[best]['roc_auc']:.3f}",
        f"Average Precision no teste: {results[best]['average_precision']:.3f}",
        "",
        "Metricas no ponto de operacao (teste):",
        f"  Sensibilidade (miopia): {tuned_metrics['recall_myopia']:.4f}  IC 95% Wilson: {_format_ci(tuned_metrics['recall_myopia_ci95'])}",
        f"  Especificidade:         {tuned_metrics['specificity']:.4f}   IC 95% Wilson: {_format_ci(tuned_metrics['specificity_ci95'])}",
        f"  Precisao (miopia):      {tuned_metrics['precision_myopia']:.4f} IC 95% Wilson: {_format_ci(tuned_metrics['precision_myopia_ci95'])}",
        f"  F1 (miopia):            {tuned_metrics['f1_myopia']:.4f}",
        f"  Acuracia balanceada:    {tuned_metrics['balanced_accuracy']:.4f}",
    ]
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Resultados da Etapa 2 (com intervalos de Wilson) salvos em %s", results_path)

    return {"best_model": best, "results": results, "fitted": fitted,
            "thresholds": thresholds, "operating_point": operating_point,
            "split": (train_idx, test_idx)}