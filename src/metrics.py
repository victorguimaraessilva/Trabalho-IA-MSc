import numpy as np
from sklearn.metrics import confusion_matrix


def sla_labels(values: np.ndarray, threshold: float) -> np.ndarray:
    """1 = conforme ao SLA (Y >= threshold), 0 = violação."""
    return (np.asarray(values) >= threshold).astype(int)


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ERR = 1 - (TP + TN) / m, conforme o enunciado da Task III."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(1 - (tp + tn) / m)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1]).ravel()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Mean Absolute Error conforme o enunciado da Task II."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = len(y_true)
    y_bar = y_true.mean()
    mae = np.abs(y_true - y_pred).sum() / m
    return float(mae / y_bar)
