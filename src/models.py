"""Feature extractors (Exp A/B), classifiers and their hyper-parameter search spaces."""
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
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


class NBSVM(BaseEstimator, ClassifierMixin):
    """NB-weighted linear classifier (Wang and Manning 2012), one-vs-rest for multi-class.

    For each class c, features are scaled by the Naive Bayes log-count ratio
    r_c = log((p_c/|p_c|_1) / (q_c/|q_c|_1)), where p_c, q_c are smoothed feature sums inside and outside c;
    a class-weighted binary Logistic Regression is then fitted on X * r_c. Prediction = argmax of the
    one-vs-rest decision scores; predict_proba is their softmax (used only for threshold tuning)."""

    def __init__(self, C: float = 4.0, alpha: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_, self.models_ = [], []
        for c in self.classes_:
            pos = y == c
            p = np.asarray(X[pos].sum(axis=0)).ravel() + self.alpha
            q = np.asarray(X[~pos].sum(axis=0)).ravel() + self.alpha
            r = np.log((p / p.sum()) / (q / q.sum()))
            clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=self.max_iter)
            self.r_.append(r)
            self.models_.append(clf.fit(X.multiply(r).tocsr(), pos))
        return self

    def decision_function(self, X):
        return np.column_stack([m.decision_function(X.multiply(r).tocsr()) for r, m in zip(self.r_, self.models_)])

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
