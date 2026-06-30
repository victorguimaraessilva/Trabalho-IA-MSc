import pandas as pd

from .config import X_PATH, Y_PATH


def load_trace() -> pd.DataFrame:
    """Carrega e une X (estatísticas do servidor) com Y (métrica de serviço)."""
    x = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)
    df = x.merge(y, on="TimeStamp", how="inner")
    if len(df) != len(x):
        raise ValueError("TimeStamps de X e Y não coincidem completamente.")
    return df


def feature_columns() -> list[str]:
    return [
        "all_..idle",
        "X..memused",
        "proc.s",
        "cswch.s",
        "file.nr",
        "sum_intr.s",
        "ldavg.1",
        "tcpsck",
        "pgfree.s",
    ]


def target_column() -> str:
    return "DispFrames"
