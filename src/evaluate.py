"""Evaluation helpers: metrics, classification report, confusion matrix, ROC/AUC."""
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy + MACRO precision/recall/F1.

    Macro = compute the metric for each class, then take the plain average.
    Every class counts equally, which suits three balanced classes.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_pipeline(pipeline, X, y, labels):
    """Predict with a fitted pipeline; return metrics, report text, confusion matrix."""
    y_pred = pipeline.predict(X)
    return {
        "metrics": compute_metrics(y, y_pred),
        "report": classification_report(y, y_pred, labels=labels, digits=3, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels),
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, labels, title="Confusion matrix (test set)", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def multiclass_roc_auc(pipeline, X, y, labels):
    """One-vs-Rest ROC/AUC. Returns (macro_auc, per_class_curves), or None if the
    model has no predict_proba (we never invent scores)."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    proba = pipeline.predict_proba(X)
    order = [list(pipeline.classes_).index(c) for c in labels]
    proba = proba[:, order]
    y_bin = label_binarize(y, classes=labels)
    macro_auc = roc_auc_score(y_bin, proba, average="macro")
    curves = {}
    for i, c in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        curves[c] = (fpr, tpr, roc_auc_score(y_bin[:, i], proba[:, i]))
    return macro_auc, curves


def plot_roc(curves, title="One-vs-Rest ROC curves (test set)", save_path=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{c} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
