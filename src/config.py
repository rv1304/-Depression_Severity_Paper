"""Central configuration: paths, seeds, class schema and shared vectorizer settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_URL = (
    "https://raw.githubusercontent.com/KUAS-ubicomp-lab/"
    "Depression_Severity_Levels_Dataset/master/Depression_Severity_Levels_Dataset.csv"
)
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "Depression_Severity_Levels_Dataset.csv"
CLEAN_CSV = DATA_DIR / "clean.csv"

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

CLASS_NAMES = ["Minimal", "Mild", "Moderate", "Severe"]
SEVERE = 3

# Accepted spellings of each label in the raw file -> ordinal class id.
LABEL_ALIASES = {
    "minimal": 0, "minimum": 0, "0": 0,
    "mild": 1, "1": 1,
    "moderate": 2, "2": 2,
    "severe": 3, "3": 3,
}

# Shared by BoW and TF-IDF so the two experiments differ only in weighting.
# token_pattern keeps single-character tokens ("i") which the default pattern drops.
VECTORIZER_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=20000,
    token_pattern=r"(?u)\b\w+\b",
)

EXPERIMENTS = {"bow": "Exp A: Bag-of-Words", "tfidf": "Exp B: TF-IDF"}
MODELS = {"nb": "Naive Bayes", "lr": "Logistic Regression", "rf": "Random Forest"}

LR_MAX_ITER = 100
# A 500-tree forest on this corpus is ~600 MB; cap concurrent RF CV fits to stay within 8 GB RAM.
RF_MAX_PARALLEL_FITS = 4
