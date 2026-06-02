import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, roc_curve
)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import REPORTS_DIR


def evaluate(y_true, y_pred, y_score=None, name="model"):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["licit", "illicit"]))
    if y_score is not None:
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)
        print(f"ROC-AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
    f1 = f1_score(y_true, y_pred)
    print(f"F1 (illicit): {f1:.4f}")
    return {"name": name, "f1": f1,
            "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else None,
            "ap": average_precision_score(y_true, y_score) if y_score is not None else None}


def plot_confusion_matrix(y_true, y_pred, name="model", save=False):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["licit", "illicit"],
                yticklabels=["licit", "illicit"], ax=ax)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(REPORTS_DIR, f"cm_{name}.png"), dpi=120)
    plt.show()


def plot_pr_curve(y_true, scores_dict, save=False):
    """scores_dict: {model_name: y_score}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_score in scores_dict.items():
        p, r, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(REPORTS_DIR, "pr_curves.png"), dpi=120)
    plt.show()


def plot_roc_curve(y_true, scores_dict, save=False):
    """scores_dict: {model_name: y_score}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(REPORTS_DIR, "roc_curves.png"), dpi=120)
    plt.show()
