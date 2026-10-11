"""Feature extractors (Exp A/B), classifiers and their hyper-parameter search spaces."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src import config


def make_vectorizer(kind: str):
    if kind == "bow":
        return CountVectorizer(**config.VECTORIZER_PARAMS)
    if kind == "tfidf":
        return TfidfVectorizer(**config.VECTORIZER_PARAMS)
    raise ValueError(f"Unknown experiment '{kind}', expected one of {list(config.EXPERIMENTS)}")


def make_model(name: str, seed: int = config.SEED):
    """Return (estimator, param_grid). Grid keys are prefixed for the 'clf' pipeline step."""
    if name == "nb":
        # Grid also selects between the Multinomial and Complement variants.
        grid = [
            {"clf": [MultinomialNB()], "clf__alpha": [0.1, 0.5, 1.0]},
            {"clf": [ComplementNB()], "clf__alpha": [0.1, 0.5, 1.0]},
        ]
        return MultinomialNB(), grid
    if name == "lr":
        # penalty='elasticnet' with l1_ratio=1.0 is pure L1; 0.5 is an even L1/L2 mix.
        est = LogisticRegression(
            solver="saga", penalty="elasticnet", l1_ratio=1.0,
            class_weight="balanced", max_iter=config.LR_MAX_ITER, tol=1e-3, random_state=seed,
        )
        grid = {"clf__C": [0.1, 1.0, 10.0], "clf__l1_ratio": [1.0, 0.5]}
        return est, grid
    if name == "rf":
        # n_jobs=1 here: parallelism is applied across CV fits instead, avoiding oversubscription.
        est = RandomForestClassifier(
            max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=seed,
        )
        grid = {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [20, 50, 100],
            "clf__min_samples_split": [2, 5, 10],
        }
        return est, grid
    raise ValueError(f"Unknown model '{name}', expected one of {list(config.MODELS)}")


def build_pipeline(experiment: str, model: str, seed: int = config.SEED):
    """Vectorizer lives inside the pipeline so it is re-fit on each CV training fold (no leakage)."""
    est, grid = make_model(model, seed)
    steps = [("vec", make_vectorizer(experiment))]
    if model == "lr" and experiment == "bow":
        # SAGA converges poorly on unscaled counts; MaxAbsScaler maps each column to [0, 1]
        # and keeps the matrix sparse. TF-IDF rows are already L2-normalised.
        steps.append(("scale", MaxAbsScaler()))
    steps.append(("clf", est))
    return Pipeline(steps), grid
