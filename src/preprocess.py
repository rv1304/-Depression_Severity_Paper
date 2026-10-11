"""Text normalisation: lowercase, contraction expansion, punctuation strip, stop words, lemmatisation."""
import re
from functools import lru_cache
from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Depression-relevant function words that generic stop lists remove.
# First-person pronouns and negation are established markers in the literature.
KEEP_WORDS = frozenset({
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "no", "not", "nor", "never", "none", "nobody", "noone", "nothing", "nowhere", "neither", "cannot",
    "alone", "empty", "against", "without",
    "cry",  # sklearn's list contains "cry"; it is an affective marker here
})
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"(?<!\w)[@/]?u/\w+|@\w+")
_SPECIAL_NEG = {"can't": "can not", "won't": "will not", "shan't": "shall not"}
_NEG = re.compile(r"n't\b")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=None)
def _wordnet_lemmatizer():
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("tests")
    except LookupError as e:
        raise RuntimeError("NLTK WordNet missing. Run: python -m nltk.downloader wordnet omw-1.4") from e
    return lemmatizer


@lru_cache(maxsize=200_000)
def wordnet_lemma(token: str) -> str:
    """Noun lemma first, then verb lemma (e.g. 'feelings' -> 'feeling', 'cried' -> 'cry')."""
    lemmatizer = _wordnet_lemmatizer()
    noun = lemmatizer.lemmatize(token, pos="n")
    return lemmatizer.lemmatize(noun, pos="v")


def clean_text(text, lemmatize: Callable[[str], str] = wordnet_lemma) -> str:
    """Return normalised token string; '' for null or content-free input."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).lower().replace("’", "'")
    t = _URL.sub(" ", t)
    t = _USER.sub(" ", t)
    for contraction, expanded in _SPECIAL_NEG.items():
        t = t.replace(contraction, expanded)
    t = _NEG.sub(" not", t)
    t = _NON_ALPHA.sub(" ", t)
    tokens = (tok for tok in _SPACES.split(t) if tok and tok not in STOP_WORDS)
    lemmas = (lemmatize(tok) for tok in tokens)
    return " ".join(lem for lem in lemmas if lem not in STOP_WORDS)


def clean_series(texts: pd.Series, lemmatize: Callable[[str], str] = wordnet_lemma) -> pd.Series:
    return texts.map(lambda t: clean_text(t, lemmatize))
