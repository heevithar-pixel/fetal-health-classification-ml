from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cardiotocography.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "cardiotocography_clean.csv"
TRAIN_PATH = BASE_DIR / "data" / "train_data.csv"
TEST_PATH = BASE_DIR / "test_data.csv"
MODEL_DIR = BASE_DIR / "model"
ARTIFACT_DIR = BASE_DIR / "artifacts"
TARGET = "NSP"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

FEATURES = [
    "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
    "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
    "Median", "Variance", "Tendency",
]

CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathologic"}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}
