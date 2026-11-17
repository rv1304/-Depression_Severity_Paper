"""DSAS depression lexicon (dataset README, after Cha, Kim and Park 2022): symptom features, lexicon-weighted
TF-IDF (Experiment C) and per-prediction symptom explanations."""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import config
from src.preprocess import clean_text

# The 11 DSAS criteria and their lexicon, verbatim from the dataset repository README.
DSAS = {
    "Depressed mood": ["sadness", "anger", "bored", "hurt", "nervousness", "moody", "emotional"],
    "Loss of interest": ["discomfort", "unhappy", "no-focus", "disturbed"],
    "Appetite change": ["loss of appetite", "overeating", "weight loss"],
    "Sleep change": ["light sleep", "waking up early", "short sleep", "dreamy"],
    "Fatigue": ["bored", "useless", "isolation", "low interaction", "fatigue"],
    "Restlessness": ["uneasiness", "instability"],
    "Anxiety": ["stress", "pain"],
    "Worthlessness": ["useless", "hopeless", "helpless", "lonely"],
    "Guilt": ["low self-esteem"],
    "Poor concentration": ["hesitate", "tired"],
    "Suicidal thoughts": ["long-term grief", "hopelessness", "end-of-my-life", "die", "suicide",
                          "killmyself", "suicidal", "broken", "worthless", "self-harm"],
}
SYMPTOMS = list(DSAS)
# Joined spellings in the README that also occur written apart in posts.
_SPLIT = {"killmyself": "kill myself"}


def _clean_term(term: str) -> str:
    """Run a lexicon entry through the same preprocessing as the posts so the two match."""
    return clean_text(_SPLIT.get(term, term).replace("-", " "))


@lru_cache(maxsize=None)
def seed_lexicon() -> dict:
    """{symptom: frozenset of preprocessed terms} for the README lexicon."""
    return {s: frozenset(t for t in map(_clean_term, terms) if t) for s, terms in DSAS.items()}


@lru_cache(maxsize=None)
def expanded_lexicon() -> dict:
    """Seed lexicon plus WordNet synonyms of every single-word entry (one- and two-word lemmas only)."""
    from nltk.corpus import wordnet as wn
    out = {}
    for symptom, terms in DSAS.items():
        found = set(seed_lexicon()[symptom])
        for term in terms:
            if "-" in term or " " in term:
                continue
            for synset in wn.synsets(term):
                for lemma in synset.lemmas():
                    c = clean_text(lemma.name().replace("_", " "))
                    if c and len(c.split()) <= 2:
                        found.add(c)
        out[symptom] = frozenset(found)
    return out


def get_lexicon(kind: str) -> dict:
    if kind == "seed":
        return seed_lexicon()
    if kind == "expanded":
        return expanded_lexicon()
    raise ValueError(f"Unknown lexicon '{kind}', expected 'seed' or 'expanded'")


def _patterns(lexicon: dict) -> list:
    # Longest terms first so "kill myself" is matched before a shorter overlapping entry.
    return [re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + r")\b")
            for terms in lexicon.values()]


def symptom_counts(texts, lexicon: dict) -> np.ndarray:
    """(n_texts, n_symptoms) raw match counts on preprocessed text."""
    pats = _patterns(lexicon)
    return np.array([[len(p.findall(t)) for p in pats] for t in texts], dtype=float).reshape(-1, len(pats))


class SymptomFeatures(BaseEstimator, TransformerMixin):
    """One log(1 + count) feature per DSAS symptom. Stateless, so it is leakage-free inside CV."""

    def __init__(self, lexicon: str = "seed"):
        self.lexicon = lexicon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sp.csr_matrix(np.log1p(symptom_counts(X, get_lexicon(self.lexicon))))


class LexiconWeightedTfidf(BaseEstimator, TransformerMixin):
    """TF-IDF whose lexicon columns (n-grams equal to a lexicon term) are multiplied by `weight`,
    then each row is re-L2-normalised. weight=1 reproduces plain TF-IDF."""

    def __init__(self, lexicon: str = "expanded", weight: float = 2.0):
        self.lexicon = lexicon
        self.weight = weight

    def fit(self, X, y=None):
        self.vec_ = TfidfVectorizer(**config.VECTORIZER_PARAMS).fit(X)
        terms = frozenset().union(*get_lexicon(self.lexicon).values())
        vocab = self.vec_.get_feature_names_out()
        self.lexicon_mask_ = np.array([v in terms for v in vocab])
        self.column_weights_ = np.where(self.lexicon_mask_, self.weight, 1.0)
        return self

    def transform(self, X):
        return normalize(self.vec_.transform(X).multiply(self.column_weights_).tocsr())


def symptom_profile(texts, labels, lexicon: dict = None) -> pd.DataFrame:
    """Share (%) of posts in each severity class that mention each symptom at least once."""
    lexicon = lexicon or seed_lexicon()
    present = symptom_counts(texts, lexicon) > 0
    labels = np.asarray(labels)
    rows = {config.CLASS_NAMES[k]: 100 * present[labels == k].mean(axis=0) for k in range(len(config.CLASS_NAMES))}
    return pd.DataFrame(rows, index=list(lexicon)).T


def term_symptom_matrix(vocab, lexicon: dict = None) -> np.ndarray:
    """(n_vocab, n_symptoms) 0/1: an n-gram belongs to a symptom when it contains one of its terms."""
    lexicon = lexicon or seed_lexicon()
    pats = _patterns(lexicon)
    return np.array([[bool(p.search(v)) for p in pats] for v in vocab], dtype=float)


def explain(pipeline, texts, lexicon: dict = None) -> pd.DataFrame:
    """Symptom-level explanation of a fitted vec->clf linear pipeline (e.g. TF-IDF + Logistic Regression).

    For each text, the contribution of a symptom to the predicted class is the sum of coefficient x feature
    value over the vocabulary n-grams that contain that symptom's terms (exact decomposition of the
    lexicon part of the linear score)."""
    lexicon = lexicon or seed_lexicon()
    vec, clf = pipeline.named_steps["vec"], pipeline.named_steps["clf"]
    X = vec.transform(texts)
    for name, step in pipeline.steps[1:-1]:  # e.g. a scaler between vectoriser and classifier
        X = step.transform(X)
    pred = clf.predict(X)
    M = term_symptom_matrix(vec.get_feature_names_out(), lexicon)
    contrib = np.vstack([X[i].multiply(clf.coef_[pred[i]]).toarray().ravel() @ M for i in range(X.shape[0])])
    out = pd.DataFrame(contrib, columns=list(lexicon))
    out.insert(0, "predicted", [config.CLASS_NAMES[p] for p in pred])
    return out
