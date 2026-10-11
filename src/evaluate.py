"""Evaluation metrics and model interpretation helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)

from src import config

LABELS = list(range(len(config.CLASS_NAMES)))


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    _, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "severe_recall": per_r[config.SEVERE],
    }
    for k, name in enumerate(config.CLASS_NAMES):
        metrics[f"f1_{name.lower()}"] = per_f1[k]
    return metrics


def normalized_confusion(y_true, y_pred) -> np.ndarray:
    """Row-normalised: cell (i, j) = share of true class i predicted as j."""
    return confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")


def report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=LABELS, target_names=config.CLASS_NAMES, digits=4, zero_division=0
    )


def lr_top_coefficients(pipeline, class_id: int = config.SEVERE, k: int = 20) -> pd.DataFrame:
    """Top-k positive and negative coefficients of one class from a fitted LR pipeline."""
    vocab = pipeline.named_steps["vec"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[class_id]
    order = np.argsort(coefs)
    idx = np.concatenate([order[:k], order[-k:]])
    df = pd.DataFrame({"feature": vocab[idx], "coefficient": coefs[idx]})
    df = df[df["coefficient"] != 0]  # L1 can zero features out; keep only active ones
    return df.drop_duplicates("feature").sort_values("coefficient").reset_index(drop=True)


def lr_sparsity(pipeline) -> float:
    """Share of vocabulary features with zero weight across every class."""
    coefs = pipeline.named_steps["clf"].coef_
    return float((np.abs(coefs).sum(axis=0) == 0).mean())


def rf_top_importances(pipeline, k: int = 20) -> pd.DataFrame:
    vocab = pipeline.named_steps["vec"].get_feature_names_out()
    imp = pipeline.named_steps["clf"].feature_importances_
    idx = np.argsort(imp)[::-1][:k]
    return pd.DataFrame({"feature": vocab[idx], "importance": imp[idx]})
