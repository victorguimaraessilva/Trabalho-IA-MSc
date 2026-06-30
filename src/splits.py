from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, SLA_THRESHOLD_FPS, TRAIN_FRACTION
from .metrics import sla_labels


def train_test_split_trace(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_fraction: float = TRAIN_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide o trace em treino e teste (amostragem aleatória uniforme)."""
    train_df, test_df = train_test_split(
        df,
        train_size=train_fraction,
        random_state=random_state,
        shuffle=True,
    )
    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()
    x_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_col].to_numpy()
    return train_df, test_df, x_train, y_train, x_test, y_test


def sample_training_subset(
    train_df: pd.DataFrame,
    n_samples: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Seleciona n observações aleatórias do conjunto de treino."""
    if n_samples > len(train_df):
        raise ValueError(f"n_samples ({n_samples}) excede o treino ({len(train_df)}).")
    return train_df.sample(n=n_samples, random_state=random_state, replace=False)


def train_test_split_sla(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float = SLA_THRESHOLD_FPS,
    train_fraction: float = TRAIN_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide o trace e retorna rótulos binários de conformidade ao SLA."""
    train_df, test_df, x_train, y_reg_train, x_test, y_reg_test = train_test_split_trace(
        df,
        feature_cols,
        target_col,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    y_train = sla_labels(y_reg_train, threshold)
    y_test = sla_labels(y_reg_test, threshold)
    return train_df, test_df, x_train, y_train, x_test, y_test, y_reg_test, y_reg_train
