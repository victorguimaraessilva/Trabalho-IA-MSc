from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
X_PATH = PROJECT_ROOT / "X.csv"
Y_PATH = PROJECT_ROOT / "Y.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# SLA: serviço conforme se taxa de frames >= 18 fps
SLA_THRESHOLD_FPS = 18

# Divisão treino/teste (Task II e III)
TRAIN_FRACTION = 0.70
RANDOM_STATE = 42

# Tamanhos dos subconjuntos de treino para estudo de acurácia
TRAIN_SIZES = [50, 500, 1000, 1500, 2520]
