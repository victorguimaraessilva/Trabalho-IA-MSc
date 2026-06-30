from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GNETTRACK_DIR = PROJECT_ROOT / "data" / "g-nettrack-pro"

# Arquivos menores com GPS e variedade de mobilidade (baixados do hackathon5G)
DEFAULT_TRACE_FILES = [
    "2023-01-21_308-walking-paulista-1.txt",
    "2023-01-22_933-driving-sp-1.txt",
    "2023-01-21_244-subway-1.txt",
]

DASH_COLUMNS = ["Longitude", "Latitude", "NetworkTech", "Accuracy", "Altitude"]
MINUS_COLUMNS = ["CQI", "SNR", "Qual", "LTERSSI"]

FEATURE_COLUMNS = ["Level", "SNR", "DL_bitrate", "UL_bitrate", "Speed"]
TARGET_COLUMN = "Qual"

FEATURE_LABELS_PT = {
    "Level": "Nível do sinal (RSRP, dBm)",
    "SNR": "Relação sinal-ruído (SNR)",
    "DL_bitrate": "Taxa de download (kbps)",
    "UL_bitrate": "Taxa de upload (kbps)",
    "Speed": "Velocidade do dispositivo (km/h)",
    "Qual": "Qualidade do sinal (Qual / RSRQ)",
}


def load_gnettrack(files: list[str] | None = None, data_dir: Path = GNETTRACK_DIR) -> pd.DataFrame:
    """Carrega e limpa arquivos tab-separados do G-NetTrack Pro."""
    paths = [data_dir / name for name in (files or DEFAULT_TRACE_FILES)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivos g-nettrack não encontrados. Execute a célula de download do notebook bônus.\n"
            + "\n".join(missing)
        )

    frames = [pd.read_csv(path, sep="\t", low_memory=False, on_bad_lines="skip") for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[df["Timestamp"] != "Timestamp"].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")

    replace_map: dict = {}
    for col in DASH_COLUMNS:
        replace_map[col] = {"--": np.nan}
    for col in MINUS_COLUMNS:
        replace_map[col] = {"-": np.nan}
    df.replace(replace_map, inplace=True)

    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "Longitude", "Latitude", "PINGAVG"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace({"Speed": {-99: np.nan}, "Height": {0: np.nan, -10000: np.nan}}, inplace=True)
    return df.sort_values("Timestamp").reset_index(drop=True)


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona linhas válidas para prever Qual a partir de métricas de rede."""
    cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    modeling = df[cols].dropna()
    return modeling.reset_index(drop=True)
