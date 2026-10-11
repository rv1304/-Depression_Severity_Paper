import numpy as np
import pandas as pd

from src.preprocess import STOP_WORDS, clean_series, clean_text


def identity(token):
    return token


def test_lowercases_and_strips_punctuation():
    assert clean_text("I feel SO tired!!!", identity) == "i feel tired"


def test_keeps_first_person_pronouns_and_negation():
    out = clean_text("I am not okay, nobody cares about me", identity).split()
    assert {"i", "not", "nobody", "me"} <= set(out)
    assert {"i", "me", "my", "not", "never"}.isdisjoint(STOP_WORDS)


def test_expands_negative_contractions():
    assert clean_text("I can't sleep and don't eat", identity) == "i not sleep not eat"
    assert "not" in clean_text("I won’t go", identity).split()  # curly apostrophe


def test_removes_urls_and_user_mentions():
    out = clean_text("https://x.com/a www.y.org @bob u/alice sad", identity)
    assert out == "sad"


def test_null_empty_and_punctuation_only_inputs():
    assert clean_text(None, identity) == ""
    assert clean_text(np.nan, identity) == ""
    assert clean_text("", identity) == ""
    assert clean_text("?!... 123", identity) == ""


def test_lemmatizer_is_applied_and_stopword_lemmas_dropped():
    lemmas = {"crying": "cry", "was": "be"}
    out = clean_text("crying was", lambda t: lemmas.get(t, t))
    assert out == "cry"


def test_clean_series_preserves_index():
    s = pd.Series(["Hopeless!", None], index=[10, 20])
    out = clean_series(s, identity)
    assert list(out.index) == [10, 20]
    assert list(out) == ["hopeless", ""]
