"""
visualization.py
================
Toda a camada visual do TCC em um so lugar. Usa backend 'Agg' (headless) para
rodar em servidores/CI sem display. Cada funcao salva um .png em outputs/figures
e devolve o caminho.

Cobre os tres pedidos de retorno visual:
  * EDA: distribuicao das classes, demografia, severidade e dioptria sintetica.
  * Antes x Depois: paineis original vs pre-processado (Etapa 1).
  * Avaliacao: matriz de confusao, curvas ROC/PR e dispersao da regressao.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem display -> nao trava em ambiente headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .labeling import SYNTHETIC_DISCLAIMER
from .utils import LOGGER

sns.set_theme(style="whitegrid", context="notebook")
_DPI = 120


def _save(fig, name: str) -> Path:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / name
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figura salva: %s", path)
    return path


def _synthetic_banner(fig) -> None:
    fig.text(0.5, -0.02, SYNTHETIC_DISCLAIMER, ha="center", va="top",
             fontsize=9, style="italic", color="firebrick")


# -----------------------------------------------------------------------------
# 1. ANALISE EXPLORATORIA (EDA)
# -----------------------------------------------------------------------------
def plot_label_distribution_full(full_df: pd.DataFrame, name="eda_01_labels_8classes.png") -> Path:
    """Distribuicao das 8 classes originais do ODIR-5K (contexto)."""
    counts = full_df["labels"].value_counts()
    counts.index = [str(i).strip("[']") for i in counts.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                palette="viridis", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontsize=9)
    ax.set(title="ODIR-5K: distribuicao das 8 classes (por imagem)",
           xlabel="Rotulo", ylabel="Nº de imagens")
    return _save(fig, name)


def plot_binary_distribution(cohort: pd.DataFrame, name="eda_02_normal_vs_miopia.png") -> Path:
    """Distribuicao Normal x Miopia + destaque do desbalanceamento."""
    counts = cohort["class_str"].map({"N": "Normal", "M": "Miopia"}).value_counts()
    ratio = counts.get("Normal", 0) / max(counts.get("Miopia", 1), 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                palette=["#4C72B0", "#C44E52"], legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set(title=f"Etapa 2 - Normal x Miopia (desbalanceamento {ratio:.1f}:1)",
           xlabel="Classe", ylabel="Nº de imagens")
    return _save(fig, name)


def plot_demographics(cohort: pd.DataFrame, name="eda_03_demografia.png") -> Path:
    """Idade (por classe) e sexo (por classe)."""
    df = cohort.copy()
    df["Classe"] = df["class_str"].map({"N": "Normal", "M": "Miopia"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=df, x="age", hue="Classe", bins=30, kde=True,
                 ax=axes[0], palette=["#4C72B0", "#C44E52"])
    axes[0].set(title="Distribuicao de idade por classe", xlabel="Idade", ylabel="Contagem")
    sns.countplot(data=df, x="sex", hue="Classe", ax=axes[1],
                  palette=["#4C72B0", "#C44E52"])
    axes[1].set(title="Distribuicao de sexo por classe", xlabel="Sexo", ylabel="Contagem")
    fig.tight_layout()
    return _save(fig, name)


def plot_severity_distribution(myopia_df: pd.DataFrame, name="eda_04_severidade.png") -> Path:
    """Distribuicao das 3 classes de severidade (ALVO SINTETICO)."""
    counts = myopia_df["severity"].value_counts().reindex(C.SEVERITY_CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                palette="rocket", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center", fontsize=11, fontweight="bold")
    ax.set(title="Etapa 3 - Severidade da miopia (leve/alta/magna)",
           xlabel="Severidade", ylabel="Nº de imagens")
    _synthetic_banner(fig)
    return _save(fig, name)


def plot_diopter_distribution(myopia_df: pd.DataFrame, name="eda_05_dioptria.png") -> Path:
    """Distribuicao do grau (dioptrias) global e por severidade (ALVO SINTETICO)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(myopia_df["diopter"], bins=25, kde=True, ax=axes[0], color="#8172B3")
    axes[0].set(title="Grau refrativo (equivalente esferico)",
                xlabel="Dioptrias (D)", ylabel="Contagem")
    sns.violinplot(data=myopia_df, x="severity", y="diopter",
                   order=C.SEVERITY_CLASSES, ax=axes[1], hue="severity",
                   palette="rocket", legend=False)
    axes[1].set(title="Grau por severidade", xlabel="Severidade", ylabel="Dioptrias (D)")
    fig.tight_layout()
    _synthetic_banner(fig)
    return _save(fig, name)


# -----------------------------------------------------------------------------
# 2. RETORNO VISUAL DE IMAGENS (Etapa 1)
# -----------------------------------------------------------------------------
def plot_preprocessing_steps(steps: dict, title: str, name: str) -> Path:
    """Mostra os estagios da Etapa 1 lado a lado para UMA imagem."""
    keys = ["1_original", "2_cropped", "3_resized", "4_clahe", "5_normalized"]
    titles = ["Original", "Recorte do fundo", "Resize 224x224",
              "CLAHE (contraste)", "Normalizada [0,1]"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.5))
    for ax, k, t in zip(axes, keys, titles):
        img = steps[k]
        ax.imshow(np.clip(img, 0, 1) if img.dtype != np.uint8 else img)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, y=1.03)
    fig.tight_layout()
    return _save(fig, name)


def plot_original_vs_preprocessed(pairs: list[tuple[np.ndarray, np.ndarray, str]],
                                  name="viz_original_vs_preproc.png") -> Path:
    """Grade original (topo) x pre-processada (base) para varias imagens."""
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, (orig, proc, label) in enumerate(pairs):
        axes[0, j].imshow(orig)
        axes[0, j].set_title(f"{label}\noriginal ({orig.shape[1]}x{orig.shape[0]})",
                             fontsize=10)
        axes[0, j].axis("off")
        axes[1, j].imshow(proc)
        axes[1, j].set_title("pre-processada (224x224)", fontsize=10)
        axes[1, j].axis("off")
    fig.tight_layout()
    return _save(fig, name)


# -----------------------------------------------------------------------------
# 3. AVALIACAO DOS MODELOS
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names: list[str],
                          title: str, name: str, synthetic: bool = False) -> Path:
    """Matriz de confusao com contagem absoluta e proporcao por linha."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    annot = np.array([f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"
                      for i in range(cm.shape[0]) for j in range(cm.shape[1])])
    annot = annot.reshape(cm.shape)
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 3,
                                    1.4 * len(class_names) + 2.5))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                vmin=0, vmax=1)
    ax.set(title=title, xlabel="Predito", ylabel="Real")
    if synthetic:
        _synthetic_banner(fig)
    return _save(fig, name)


def plot_roc_curves(curves: dict[str, dict], name="stage2_roc.png") -> Path:
    """Sobrepoe curvas ROC de varios modelos (Etapa 2)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], lw=2, label=f"{label} (AUC={c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    ax.set(title="Etapa 2 - Curva ROC", xlabel="Taxa de falso positivo",
           ylabel="Taxa de verdadeiro positivo", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, name)


def plot_pr_curves(curves: dict[str, dict], name="stage2_pr.png") -> Path:
    """Curvas Precision-Recall (mais informativas sob desbalanceamento)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, c in curves.items():
        ax.plot(c["recall"], c["precision"], lw=2,
                label=f"{label} (AP={c['ap']:.3f})")
    ax.set(title="Etapa 2 - Curva Precision-Recall", xlabel="Recall (Miopia)",
           ylabel="Precision (Miopia)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, name)


def plot_threshold_analysis(y_true, y_score, chosen_threshold: float,
                            target_sensitivity: float, title: str,
                            name="stage2_threshold.png") -> Path:
    """
    Sensibilidade (recall da miopia) e precisao em funcao do limiar de decisao,
    com o alvo de sensibilidade e o limiar escolhido destacados. As curvas usam
    os escores do TESTE (ilustra a generalizacao do ponto de operacao); o limiar
    em si foi escolhido sobre escores out-of-fold do treino (sem vazamento).
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]  # alinha com `thr` (ultimo par nao tem limiar)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(thr, rec, lw=2, color="#C44E52", label="Sensibilidade (recall Miopia)")
    ax.plot(thr, prec, lw=2, color="#4C72B0", label="Precisao (Miopia)")
    ax.axhline(target_sensitivity, ls=":", lw=1.2, color="#C44E52",
               label=f"Alvo de sensibilidade = {target_sensitivity:.2f}")
    ax.axvline(chosen_threshold, ls="--", lw=1.6, color="k",
               label=f"Limiar escolhido = {chosen_threshold:.3f}")
    ax.set(title=title, xlabel="Limiar de decisao", ylabel="Metrica",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower center", fontsize=9)
    return _save(fig, name)


def plot_model_comparison(scores: dict[str, dict], metric: str,
                          title: str, name: str) -> Path:
    """Barra comparando um metrica entre modelos."""
    labels = list(scores.keys())
    values = [scores[m][metric] for m in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=labels, y=values, ax=ax, hue=labels, palette="mako", legend=False)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set(title=title, xlabel="Modelo", ylabel=metric, ylim=(0, 1.05))
    return _save(fig, name)


def plot_feature_importance(names: list[str], importances: np.ndarray,
                            top_k: int, title: str, name: str) -> Path:
    """Top-k features mais importantes (Random Forest)."""
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_k + 1.5))
    sns.barplot(x=importances[order], y=[names[i] for i in order],
                ax=ax, hue=[names[i] for i in order], palette="flare", legend=False)
    ax.set(title=title, xlabel="Importancia", ylabel="")
    fig.tight_layout()
    return _save(fig, name)


def plot_regression_scatter(y_true, y_pred, r2: float, mae: float,
                            name="stage4_scatter.png") -> Path:
    """Dispersao grau real x grau predito (ALVO SINTETICO)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lo = min(y_true.min(), y_pred.min()) - 1
    hi = max(y_true.max(), y_pred.max()) + 1
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3, color="#8172B3")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal (y=x)")
    ax.set(title=f"Etapa 4 - Grau real x predito (R²={r2:.3f}, MAE={mae:.2f} D)",
           xlabel="Grau real (D)", ylabel="Grau predito (D)",
           xlim=(lo, hi), ylim=(lo, hi))
    ax.legend(loc="upper left")
    _synthetic_banner(fig)
    return _save(fig, name)


def plot_residuals(y_true, y_pred, name="stage4_residuos.png") -> Path:
    """Residuos (real - predito) x predito, para inspecionar vies/heterocedasticidade."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, resid, alpha=0.6, edgecolor="k", linewidth=0.3, color="#55A868")
    axes[0].axhline(0, color="r", ls="--", lw=1.5)
    axes[0].set(title="Residuos x predito", xlabel="Grau predito (D)", ylabel="Residuo (D)")
    sns.histplot(resid, bins=25, kde=True, ax=axes[1], color="#55A868")
    axes[1].set(title="Distribuicao dos residuos", xlabel="Residuo (D)", ylabel="Contagem")
    fig.tight_layout()
    _synthetic_banner(fig)
    return _save(fig, name)