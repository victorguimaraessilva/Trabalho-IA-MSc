"""Rótulos em português para tabelas e relatório."""

from __future__ import annotations

import pandas as pd

FEATURE_LABELS_PT: dict[str, str] = {
    "all_..idle": "CPU ociosa (%)",
    "X..memused": "Memória usada (%)",
    "proc.s": "Taxa de criação de processos (/s)",
    "cswch.s": "Taxa de troca de contexto (/s)",
    "file.nr": "File handles em uso",
    "sum_intr.s": "Taxa de interrupções (/s)",
    "ldavg.1": "Média de carga — 1 min",
    "tcpsck": "Sockets TCP em uso",
    "pgfree.s": "Taxa de liberação de páginas (/s)",
}

TARGET_LABEL_PT = "Taxa de frames de vídeo (fps)"
TIMESTAMP_LABEL_PT = "Carimbo de tempo (Unix)"

STAT_ROW_LABELS_PT: dict[str, str] = {
    "mean": "Média",
    "max": "Máximo",
    "min": "Mínimo",
    "p25": "Percentil 25",
    "p90": "Percentil 90",
    "std": "Desvio padrão",
}

REGRESSION_RESULTS_COLS_PT: dict[str, str] = {
    "n_train": "Observações no treino",
    "train_time_ms": "Tempo de treino (ms)",
    "nmae": "NMAE (proporção)",
    "nmae_pct": "NMAE (%)",
    "accurate": "Acurado (< 15%)?",
}

CLASSIFICATION_RESULTS_COLS_PT: dict[str, str] = {
    "classifier": "Classificador",
    "n_train": "Observações no treino",
    "train_time_ms": "Tempo de treino (ms)",
    "err": "Erro de classificação (ERR)",
    "err_pct": "ERR (%)",
    "accurate": "Acurado (< 15%)?",
    "TP": "Verdadeiros positivos (VP)",
    "TN": "Verdadeiros negativos (VN)",
    "FP": "Falsos positivos (FP)",
    "FN": "Falsos negativos (FN)",
}

COEF_COLS_PT: dict[str, str] = {
    "feature": "Variável",
    "theta": "Coeficiente (Θ)",
    "intercept": "Intercepto (Θ₀)",
    "intercept (theta_0)": "Intercepto (Θ₀)",
}


def feature_label(name: str) -> str:
    if name == "DispFrames":
        return TARGET_LABEL_PT
    if name == "TimeStamp":
        return TIMESTAMP_LABEL_PT
    if name in ("intercept", "intercept (theta_0)"):
        return COEF_COLS_PT.get(name, name)
    return FEATURE_LABELS_PT.get(name, name)


def rename_trace_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {**FEATURE_LABELS_PT, "DispFrames": TARGET_LABEL_PT, "TimeStamp": TIMESTAMP_LABEL_PT}
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def rename_stat_table(stats: pd.DataFrame) -> pd.DataFrame:
    renamed = stats.rename(columns=FEATURE_LABELS_PT | {"DispFrames": TARGET_LABEL_PT})
    return renamed.rename(index=STAT_ROW_LABELS_PT)


def _format_yes_no(df: pd.DataFrame, column: str = "accurate") -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = df[column].map({True: "Sim", False: "Não"})
    return df


def rename_regression_results(df: pd.DataFrame) -> pd.DataFrame:
    return _format_yes_no(df).rename(columns=REGRESSION_RESULTS_COLS_PT)


def rename_classification_results(df: pd.DataFrame) -> pd.DataFrame:
    return _format_yes_no(df).rename(columns=CLASSIFICATION_RESULTS_COLS_PT)


def rename_coefficient_table(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {col: feature_label(col) for col in df.columns}
    if "n_train" in df.columns:
        col_map["n_train"] = "Observações no treino"
    out = df.rename(columns=col_map)

    if out.index.name == "n_train":
        out.index.name = "Observações no treino"
    elif out.index.name == "classifier":
        out.index.name = "Classificador"
    return out


def format_model_coefficients(features: list[str], coefs, intercept: float) -> pd.DataFrame:
    rows = [{"Variável": feature_label(f), "Coeficiente (Θ)": c} for f, c in zip(features, coefs)]
    rows.append({"Variável": "Intercepto (Θ₀)", "Coeficiente (Θ)": intercept})
    return pd.DataFrame(rows)
