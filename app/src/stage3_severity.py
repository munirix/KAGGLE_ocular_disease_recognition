"""
stage3_severity.py -- ETAPA 3 (DEMONSTRACAO)
===========================================
Classificacao de "severidade" em K classes ordinais definidas por C.SEVERITY_CLASSES.
Roda apenas sobre a coorte de miopia. Os rotulos sao um PROXY SINTETICO derivado
do MDI (ver labeling.py). Esta etapa e uma demonstracao arquitetural: nao
produz evidencia clinica de graduacao de miopia.

O split e a validacao cruzada sao agrupados por paciente (patient_id) para
evitar vazamento entre os olhos. A selecao do melhor modelo usa F1-macro medio
de CV, nao o teste.

Os rotulos devem ser gerados antes (tipicamente por pipeline.prepare_myopia)
usando estatisticas apenas do treino; caso contrario, esta etapa os gera a
partir de todo o conjunto para permitir execucao autonoma (com aviso).
"""
from __future__ import annotations

import joblib
import warnings
import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import labeling, visualization as viz
from .logging_utils import LOGGER
from .provenance import build_manifest, file_sha256
from .utils import save_json, timer

_SEV_TO_INT = {s: i for i, s in enumerate(C.SEVERITY_CLASSES)}


class _LogisticATCompat(LogisticAT):
    """Wrapper compativel para o LogisticAT do mord 0.7.

    Resolve duas incompatibilidades:
      1. O mord 0.7 passa a opcao `disp` para o L-BFGS-B do SciPy, o que gera
         DeprecationWarning a partir do SciPy 1.15.
      2. O LogisticAT exige que as classes em y sejam 0..K-1 consecutivas;
         em subamostras pequenas algumas classes podem estar ausentes.

    Ate que haja uma versao nova do mord, silenciamos o aviso e re-mapeamos
    as classes para indices consecutivos, restaurando os rotulos originais em
    `predict`.
    """

    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            y_arr = np.asarray(y)
            self._orig_classes_ = np.unique(y_arr)
            self._label_map_ = {i: int(c) for i, c in enumerate(self._orig_classes_)}
            self._inv_label_map_ = {int(c): i for i, c in enumerate(self._orig_classes_)}
            y_remapped = np.array([self._inv_label_map_[v] for v in y_arr], dtype=int)
            super().fit(X, y_remapped, sample_weight=sample_weight)
            # O super().fit redefine self.classes_; mantemos o mapeamento original.
            return self

    def predict(self, X):
        pred = super().predict(X)
        return np.array([self._label_map_[p] for p in pred], dtype=int)


def encode_severity(severity: np.ndarray) -> np.ndarray:
    """Converte rotulos textuais (ordem crescente de gravidade) em inteiros ordinais 0..K-1."""
    return np.array([_SEV_TO_INT[s] for s in severity], dtype=int)


def _build_models(cfg: C.RunConfig) -> dict[str, Pipeline]:
    p = C.STAGE3
    cw = p.use_class_weight
    return {
        "ordinal_logistic": Pipeline([
            ("scaler", StandardScaler()),
            # LogisticAT (All-Threshold) do mord: regressao logistica ordinal
            # que respeita a ordenacao leve < moderado < alta < magna.
            ("clf", _LogisticATCompat(alpha=p.ordinal_alpha, max_iter=5000, verbose=0)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=p.rf_n_estimators, max_depth=p.rf_max_depth,
                min_samples_leaf=p.rf_min_samples_leaf, class_weight=cw,
                n_jobs=cfg.n_jobs, random_state=cfg.seed)),
        ]),
    }


def _f1_macro_scorer(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """Scorer customizado para F1-macro sobre todas as classes de severidade."""
    y_pred = estimator.predict(X)
    return float(f1_score(
        y, y_pred, labels=list(range(len(C.SEVERITY_CLASSES))),
        average="macro", zero_division=0))


def _max_splits_by_class(y: np.ndarray, groups: np.ndarray) -> int:
    """Numero maximo de dobras para StratifiedGroupKFold dado o minimo de grupos/classe."""
    counts = {}
    for yi, gi in zip(y, groups):
        counts.setdefault(yi, set()).add(gi)
    return min(len(s) for s in counts.values())


def _split_grouped(X: np.ndarray, groups: np.ndarray,
                   cfg: C.RunConfig,
                   y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split agrupado por paciente. Usa StratifiedGroupKFold quando houver grupos suficientes."""
    if y is not None:
        max_splits = _max_splits_by_class(y, groups)
        if max_splits >= 2 * C.CV_FOLDS:
            gss = StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True,
                                       random_state=cfg.seed)
            return next(gss.split(X, y, groups))
        LOGGER.warning("Numero insuficiente de grupos por classe para StratifiedGroupKFold; "
                       "usando GroupShuffleSplit.")

    n_unique = len(np.unique(groups))
    if n_unique < 2:
        raise ValueError("Nao e possivel dividir: menos de 2 grupos de pacientes.")
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=cfg.seed)
    return next(gss.split(X, groups=groups))


def run_stage3(X: np.ndarray, myopia_df: pd.DataFrame, feature_names: list[str],
               cfg: C.RunConfig | None = None,
               train_idx: np.ndarray | None = None,
               test_idx: np.ndarray | None = None,
               command: str = "stage3") -> dict:
    """Treina/avalia a classificacao de severidade (demonstracao) e gera figuras."""
    cfg = cfg or C.RunConfig()
    groups = myopia_df[C.GROUP_COL].to_numpy()

    # Se os rotulos ainda nao existirem, divide primeiro por paciente e depois
    # gera os alvos apenas no treino, evitando vazamento mesmo em execucao autonoma.
    if "severity" not in myopia_df.columns:
        if train_idx is None or test_idx is None:
            train_idx, test_idx = _split_grouped(X, groups, cfg)
        myopia_df = labeling.assign_synthetic_targets(myopia_df, cfg, fit_idx=train_idx)

    y = encode_severity(myopia_df["severity"].to_numpy())
    filenames = myopia_df["filename"].to_numpy()
    patient_ids = myopia_df[C.GROUP_COL].to_numpy()
    LOGGER.info("=== ETAPA 3: Severidade (DEMO) | X=%s | classes=%s | pacientes=%d ===",
                X.shape, np.bincount(y).tolist(), len(np.unique(groups)))

    # Split agrupado por paciente (~C.TEST_SIZE de teste).
    if train_idx is None or test_idx is None:
        train_idx, test_idx = _split_grouped(X, groups, cfg, y=y)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = groups[train_idx]

    max_splits = _max_splits_by_class(y_tr, g_tr)
    # Garante pelo menos ~2 grupos da classe minoritaria por fold (evita folds degenerados).
    cv_splits = min(C.CV_FOLDS, max(2, max_splits // 2))
    if max_splits >= 2 and cv_splits == max_splits:
        cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=cfg.seed)
    else:
        n_unique_tr = len(np.unique(g_tr))
        cv_splits = max(2, min(C.CV_FOLDS, n_unique_tr))
        LOGGER.warning("Numero insuficiente de grupos por classe para CV estratificada; "
                       "usando GroupKFold (%d dobras).", cv_splits)
        cv = GroupKFold(n_splits=cv_splits)

    results, fitted = {}, {}
    for name, model in _build_models(cfg).items():
        with timer(f"Etapa3:{name}"):
            cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, groups=g_tr,
                                    scoring=_f1_macro_scorer, n_jobs=cfg.n_jobs)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results[name] = {
                "cv_f1_macro_mean": float(cv_f1.mean()),
                "cv_f1_macro_std": float(cv_f1.std()),
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "balanced_accuracy": float(recall_score(
                    y_te, y_pred, labels=list(range(len(C.SEVERITY_CLASSES))),
                    average="macro", zero_division=0)),
                "f1_macro": float(f1_score(
                    y_te, y_pred, labels=list(range(len(C.SEVERITY_CLASSES))),
                    average="macro", zero_division=0)),
                "report": classification_report(
                    y_te, y_pred, labels=list(range(len(C.SEVERITY_CLASSES))),
                    target_names=C.SEVERITY_CLASSES,
                    output_dict=True, zero_division=0),
            }
            fitted[name] = model
            LOGGER.info("%s -> CV-F1_macro=%.3f test-F1_macro=%.3f bal_acc=%.3f",
                        name, results[name]["cv_f1_macro_mean"],
                        results[name]["f1_macro"], results[name]["balanced_accuracy"])

    best = max(results, key=lambda m: results[m]["cv_f1_macro_mean"])
    LOGGER.info("Melhor modelo Etapa 3 (DEMO): %s (CV-F1_macro=%.3f | test=%.3f)",
                best, results[best]["cv_f1_macro_mean"], results[best]["f1_macro"])

    y_pred_best = fitted[best].predict(X_te)
    viz.plot_confusion_matrix(y_te, y_pred_best, C.SEVERITY_CLASSES,
                              f"Etapa 3 - Severidade ({best}) - DEMONSTRACAO SINTETICA",
                              "stage3_confusion.png", synthetic=True)
    viz.plot_model_comparison(results, "cv_f1_macro_mean",
                              "Etapa 3 - F1-macro por modelo (CV agrupada)",
                              "stage3_f1_comparacao.png", synthetic=True)
    if "random_forest" in fitted:
        rf = fitted["random_forest"].named_steps["clf"]
        viz.plot_feature_importance(feature_names, rf.feature_importances_, 20,
                                    "Etapa 3 - Top 20 features (Random Forest)",
                                    "stage3_feature_importance.png", synthetic=True)

    joblib.dump(fitted[best], C.MODELS_DIR / "stage3_best_model.joblib")

    # Predicoes por imagem (OOF no treino, modelo final no teste).
    oof_pred = cross_val_predict(fitted[best], X_tr, y_tr, cv=cv, groups=g_tr,
                                 n_jobs=cfg.n_jobs, method="predict")
    severity_true = myopia_df["severity"].to_numpy()
    preds_rows = []
    for j, i in enumerate(train_idx):
        preds_rows.append({
            "filename": str(filenames[i]),
            "patient_id": str(patient_ids[i]),
            "partition": "train",
            "severity_true": str(severity_true[i]),
            "severity_pred": str(C.SEVERITY_CLASSES[int(oof_pred[j])]),
            "model": best,
        })
    for j, i in enumerate(test_idx):
        preds_rows.append({
            "filename": str(filenames[i]),
            "patient_id": str(patient_ids[i]),
            "partition": "test",
            "severity_true": str(severity_true[i]),
            "severity_pred": str(C.SEVERITY_CLASSES[int(y_pred_best[j])]),
            "model": best,
        })
    C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = C.PREDICTIONS_DIR / "stage3_predictions.csv"
    pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
    LOGGER.info("Predicoes da Etapa 3 salvas em %s", preds_path)

    split_summary = {
        "n_train_images": int(len(train_idx)),
        "n_test_images": int(len(test_idx)),
        "n_train_patients": int(len(set(groups[train_idx]))),
        "n_test_patients": int(len(set(groups[test_idx]))),
        "n_overlap_patients": int(len(set(groups[train_idx]) & set(groups[test_idx]))),
    }

    payload = {
        "best_model": best,
        "best_criterion": "cv_f1_macro_mean",
        "warning": "DEMONSTRACAO COM ALVO SINTETICO - SEM VALIDADE CLINICA",
        "results": results,
        "split_summary": split_summary,
    }
    save_json(payload, C.REPORTS_DIR / "stage3_metrics.json")

    manifest = build_manifest(cfg, command=command,
                              extra={"stage": "stage3_demo",
                                     "data_file_hash": file_sha256(C.FULL_DF_CSV),
                                     "cohort_size": len(y),
                                     "split_summary": split_summary})
    save_json(manifest, C.REPORTS_DIR / "stage3_manifest.json")

    # Secao de resultados legivel (estilo stage2_results.txt)
    res = results[best]
    report = res["report"]
    sev_label = " / ".join(C.SEVERITY_CLASSES)
    col_width = max(len(c) for c in C.SEVERITY_CLASSES)
    lines = [
        f"=== Resultados da Etapa 3 (Severidade: {sev_label}) - DEMONSTRACAO ===",
        "Aviso: alvos sinteticos - sem validade clinica",
        f"Melhor modelo: {best}",
        f"Particao: {split_summary['n_train_images']} treino / {split_summary['n_test_images']} teste",
        f"Pacientes em comum: {split_summary['n_overlap_patients']}",
        f"CV F1-macro: {res['cv_f1_macro_mean']:.3f} (+/- {res['cv_f1_macro_std']:.3f})",
        "",
        "Metricas no teste:",
        f"  Acuracia:            {res['accuracy']:.4f}",
        f"  Acuracia balanceada: {res['balanced_accuracy']:.4f}",
        f"  F1-macro:            {res['f1_macro']:.4f}",
        "",
        "Relatorio por classe (teste):",
    ]
    for cls in C.SEVERITY_CLASSES:
        r = report[cls]
        lines.append(
            f"  {cls:<{col_width}} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"F1={r['f1-score']:.4f} support={int(r['support'])}"
        )
    results_path = C.REPORTS_DIR / "stage3_results.txt"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Resultados da Etapa 3 salvos em %s", results_path)

    return {"best_model": best, "results": results, "fitted": fitted,
            "split_summary": split_summary}