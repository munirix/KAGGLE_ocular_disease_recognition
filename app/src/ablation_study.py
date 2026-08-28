"""
ablation_study.py -- Estudo de ablacao da Etapa 4
=================================================
A Etapa 4 prediz um equivalente esferico SINTETICO (diopter) derivado do MDI.
Como o MDI e calculado a partir de medidas da propria imagem, parte dos 169
atributos pode ser correlacionada com o alvo. Este modulo quantifica ate que
ponto o desempenho do regressor depende desses atributos, removendo os *k*
atributos mais correlacionados com o alvo (calculados apenas no treino) e
retreinando os modelos.

O estudo e executado como experimento PARALELO: os artefatos da Etapa 4
(`stage4_metrics.json`, `stage4_results.txt`, figuras `stage4_*.png`) NAO sao
sobrescritos. Os resultados da ablacao sao salvos com o prefixo
`stage4_ablation_*`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_score

from . import config as C
from . import pipeline
from . import stage4_regression as s4
from . import visualization as viz
from .logging_utils import LOGGER
from .provenance import build_manifest, file_sha256
from .utils import save_json, timer


def _correlation_ranking(X: np.ndarray, y: np.ndarray,
                        feature_names: list[str]) -> pd.Series:
    """Retorna atributos ordenados pelo valor absoluto da correlacao com y."""
    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0, keepdims=True)
    num = X_centered.T @ y_centered
    den = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
    den[den == 0] = 1.0
    corr = num / den
    return pd.Series(np.abs(corr), index=feature_names).sort_values(ascending=False)


def _cv_splits(groups: np.ndarray, cfg: C.RunConfig):
    """Reutiliza a mesma CV agrupada da Etapa 4."""
    n_unique_tr = len(np.unique(groups))
    cv_splits = max(2, min(C.CV_FOLDS, n_unique_tr // 2))
    return GroupKFold(n_splits=cv_splits)


def _run_single_ablation(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        feature_names: list[str], train_idx: np.ndarray,
                        test_idx: np.ndarray, cfg: C.RunConfig,
                        k: int) -> dict:
    """Executa a Etapa 4 removendo os `k` atributos mais correlacionados do treino."""
    X_tr, y_tr = X[train_idx], y[train_idx]

    ranking = _correlation_ranking(X_tr, y_tr, feature_names)
    removed = ranking.head(k).index.tolist()
    keep_mask = ~np.isin(feature_names, removed)
    X_ablated = X[:, keep_mask]
    X_ab_tr, X_ab_te = X_ablated[train_idx], X_ablated[test_idx]
    y_te = y[test_idx]
    g_tr = groups[train_idx]

    results: dict[str, dict] = {}
    cv = _cv_splits(g_tr, cfg)
    for name, model in s4._build_models(cfg).items():
        with timer(f"Etapa4_ablation(k={k}):{name}"):
            cv_r2 = cross_val_score(model, X_ab_tr, y_tr, cv=cv, groups=g_tr,
                                    scoring=s4._r2_scorer, n_jobs=cfg.n_jobs)
            cv_mae = -cross_val_score(model, X_ab_tr, y_tr, cv=cv, groups=g_tr,
                                      scoring="neg_mean_absolute_error",
                                      n_jobs=cfg.n_jobs)

            model.fit(X_ab_tr, y_tr)
            y_pred = model.predict(X_ab_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_r2 = float(r2_score(y_te, y_pred))
            results[name] = {
                "cv_r2_mean": float(cv_r2.mean()),
                "cv_mae_mean": float(cv_mae.mean()),
                "r2": test_r2,
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "rmse": float(root_mean_squared_error(y_te, y_pred)),
                "preds": y_pred.tolist(),
            }

    best = max(results, key=lambda m: results[m]["cv_r2_mean"])
    return {
        "k": k,
        "n_features_total": int(X.shape[1]),
        "n_features_removed": int(k),
        "n_features_kept": int(X_ablated.shape[1]),
        "best_model": best,
        "results": results,
        "removed_features": removed,
        "ranking": ranking.head(k).to_dict(),
    }


def _plot_ablation_curve(ablations: list[dict], baseline: dict, output_name: str,
                        metric: str, ylabel: str, title: str) -> Path:
    """Plota metrica de desempenho vs numero de atributos removidos."""
    ks = [a["k"] for a in ablations]
    values = [a["results"][a["best_model"]][metric] for a in ablations]
    baseline_value = baseline["results"][baseline["best_model"]][metric]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(baseline_value, color="red", linestyle="--",
               label=f"Baseline (0 removidos): {baseline_value:.3f}")
    ax.plot(ks, values, marker="o", color="steelblue", label=f"Ablacao ({metric})")
    ax.set_xlabel("Atributos removidos (top-k correlacionados com diopter)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    viz._synthetic_banner(fig)
    return viz._save(fig, output_name)


def run_ablation(cfg: C.RunConfig | None = None,
                ks: list[int] | None = None) -> dict:
    """Executa o estudo de ablacao completo e persiste artefatos separados.

    Parametros
    ----------
    cfg : RunConfig
        Configuracao de execucao.
    ks : list[int]
        Valores de k (numero de atributos a remover) a testar.
        Padrao: [5, 10, 20, 50].
    """
    cfg = cfg or C.RunConfig()
    ks = ks or [5, 10, 20, 50]

    # Reutiliza a preparacao existente para garantir comparabilidade com baseline.
    _, cohort, X, feature_names = pipeline.prepare(cfg)
    X_my, myopia_df, train_idx, test_idx = pipeline.prepare_myopia(cfg, cohort, X)

    if "diopter" not in myopia_df.columns:
        from . import labeling
        myopia_df = labeling.assign_synthetic_targets(myopia_df, cfg, fit_idx=train_idx)

    y = myopia_df["diopter"].to_numpy(dtype=float)
    groups = myopia_df[C.GROUP_COL].to_numpy()

    LOGGER.info("=== Estudo de ablacao da Etapa 4 | X=%s | ks=%s ===", X_my.shape, ks)

    # Executa baseline sem remocao para comparacao no mesmo split.
    # persist=False evita sobrescrever os artefatos stage4_*.png/json/txt originais.
    baseline_result = s4.run_stage4(X_my, myopia_df, feature_names, cfg,
                                   train_idx=train_idx, test_idx=test_idx,
                                   command="stage4_baseline_reexec",
                                   persist=False)

    # Roda ablacoes para cada k.
    ablations = []
    for k in ks:
        if k >= X_my.shape[1]:
            LOGGER.warning("k=%d maior ou igual ao numero de atributos (%d); pulando.",
                           k, X_my.shape[1])
            continue
        ab = _run_single_ablation(X_my, y, groups, feature_names, train_idx,
                                  test_idx, cfg, k)
        ablations.append(ab)
        LOGGER.info(
            "Ablacao k=%d: melhor=%s | R2=%.3f | MAE=%.2f D | removidos=%s",
            ab["k"], ab["best_model"],
            ab["results"][ab["best_model"]]["r2"],
            ab["results"][ab["best_model"]]["mae"],
            ", ".join(ab["removed_features"][:5]) + ("..." if len(ab["removed_features"]) > 5 else "")
        )

    # Salva metricas consolidadas.
    payload = {
        "warning": "DEMONSTRACAO COM ALVO SINTETICO - ABLACAO POR CORRELACAO",
        "baseline": {
            "best_model": baseline_result["best_model"],
            "results": baseline_result["results"],
            "n_features": int(X_my.shape[1]),
        },
        "ablations": {str(a["k"]): a for a in ablations},
        "feature_ranking": _correlation_ranking(X_my[train_idx], y[train_idx],
                                                feature_names).head(50).to_dict(),
    }
    metrics_path = C.REPORTS_DIR / "stage4_ablation_metrics.json"
    save_json(payload, metrics_path)

    # Salva lista de atributos removidos para consulta.
    removed_path = C.REPORTS_DIR / "stage4_ablation_removed_features.csv"
    removed_rows = []
    for a in ablations:
        for rank, feat in enumerate(a["removed_features"], start=1):
            removed_rows.append({"k": a["k"], "rank": rank, "feature": feat,
                                 "abs_correlation": a["ranking"].get(feat, 0.0)})
    pd.DataFrame(removed_rows).to_csv(removed_path, index=False)

    # Figuras comparativas.
    _plot_ablation_curve(ablations, payload["baseline"],
                        "stage4_ablation_r2_by_k.png", "r2",
                        "R2 no teste", "Etapa 4 - R2 vs atributos removidos")
    _plot_ablation_curve(ablations, payload["baseline"],
                        "stage4_ablation_mae_by_k.png", "mae",
                        "MAE no teste (D)", "Etapa 4 - MAE vs atributos removidos")

    # Scatter do melhor modelo ablacionado para k=5 (simbolico: 5 componentes do MDI).
    if ablations:
        k5 = next((a for a in ablations if a["k"] == 5), ablations[0])
        viz.plot_regression_scatter(
            y[test_idx],
            np.asarray(k5["results"][k5["best_model"]]["preds"]),
            k5["results"][k5["best_model"]]["r2"],
            k5["results"][k5["best_model"]]["mae"],
            name="stage4_ablation_scatter_k5.png"
        )

    # Resultados legiveis.
    lines = [
        "=== Estudo de ablacao da Etapa 4 (alvo sintetico) ===",
        "Aviso: alvos sinteticos - sem validade clinica",
        f"Atributos totais: {X_my.shape[1]}",
        "",
        "Baseline (sem remocao):",
        f"  Melhor modelo: {payload['baseline']['best_model']}",
    ]
    bm = payload["baseline"]["best_model"]
    br = payload["baseline"]["results"][bm]
    lines += [
        f"  R2:   {br['r2']:.4f}",
        f"  MAE:  {br['mae']:.2f} D",
        f"  RMSE: {br['rmse']:.2f} D",
        "",
        "Ablacao (top-k atributos mais correlacionados removidos no treino):",
    ]
    for a in ablations:
        am = a["best_model"]
        ar = a["results"][am]
        lines += [
            f"k={a['k']} (mantidos {a['n_features_kept']}):",
            f"  Melhor modelo: {am}",
            f"  R2: {ar['r2']:.4f} | MAE: {ar['mae']:.2f} D | RMSE: {ar['rmse']:.2f} D",
        ]
    results_path = C.REPORTS_DIR / "stage4_ablation_results.txt"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Manifesto.
    manifest = build_manifest(
        cfg, command="ablation",
        extra={"stage": "stage4_ablation",
               "data_file_hash": file_sha256(C.FULL_DF_CSV),
               "ks": ks,
               "baseline_best": payload["baseline"]["best_model"],
               "n_features": int(X_my.shape[1])}
    )
    save_json(manifest, C.REPORTS_DIR / "stage4_ablation_manifest.json")

    LOGGER.info("Estudo de ablacao concluido. Relatorio: %s", metrics_path)
    return payload


if __name__ == "__main__":
    from .logging_utils import setup_tee_logging
    C.ensure_dirs()
    import datetime
    log_path = C.OUTPUTS_DIR / f"ablation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_tee_logging(log_path)
    run_ablation()