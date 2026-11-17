import numpy as np
import pytest

from src import config, evaluate, extensions, lexicon
from src.models import NBSVM
from tests.test_models_and_funnel import toy_corpus


def test_seed_lexicon_matches_preprocessed_posts():
    lex = lexicon.seed_lexicon()
    assert list(lex) == lexicon.SYMPTOMS and len(lex) == 11
    assert "kill myself" in lex["Suicidal thoughts"]  # README spelling "killmyself" split
    assert "end my life" in lex["Suicidal thoughts"]  # hyphens removed, "of" is a stop word
    assert all(terms for terms in lex.values())


def test_expanded_lexicon_is_superset_of_seed():
    seed, exp = lexicon.seed_lexicon(), lexicon.expanded_lexicon()
    assert all(seed[s] <= exp[s] for s in seed)
    assert sum(map(len, exp.values())) > sum(map(len, seed.values()))


def test_symptom_counts_whole_words_only():
    counts = lexicon.symptom_counts(["i want to die die", "diet plan", ""], lexicon.seed_lexicon())
    suicidal = lexicon.SYMPTOMS.index("Suicidal thoughts")
    assert counts.shape == (3, 11)
    assert counts[0, suicidal] == 2
    assert counts[1].sum() == 0 and counts[2].sum() == 0


def test_symptom_features_are_log_counts():
    X = lexicon.SymptomFeatures().fit_transform(["hopeless lonely", "fine"])
    worth = lexicon.SYMPTOMS.index("Worthlessness")
    assert X.shape == (2, 11)
    assert np.isclose(X[0, worth], np.log1p(2)) and X[1].sum() == 0


def test_lexicon_weighted_tfidf_weight_one_is_plain_tfidf(monkeypatch):
    monkeypatch.setitem(config.VECTORIZER_PARAMS, "min_df", 1)
    X, _ = toy_corpus(5)
    plain = lexicon.LexiconWeightedTfidf(weight=1.0).fit(X)
    heavy = lexicon.LexiconWeightedTfidf(weight=3.0).fit(X)
    assert np.allclose(plain.transform(X).toarray(), plain.vec_.transform(X).toarray())
    assert heavy.lexicon_mask_.any()
    rows = np.sqrt(np.asarray(heavy.transform(X).multiply(heavy.transform(X)).sum(axis=1))).ravel()
    assert np.allclose(rows, 1.0)  # rows stay L2-normalised


def test_symptom_profile_shares():
    prof = lexicon.symptom_profile(["i want to die", "fine day", "lonely", "die"], [3, 0, 1, 3])
    assert prof.loc["Severe", "Suicidal thoughts"] == 100
    assert prof.loc["Minimal"].sum() == 0
    assert list(prof.index) == config.CLASS_NAMES


def test_qwk_penalises_far_errors_more():
    y = np.array([0, 1, 2, 3] * 5)
    near = evaluate.compute_metrics(y, np.clip(y + 1, 0, 3))["qwk"]
    far = evaluate.compute_metrics(y, 3 - y)["qwk"]
    assert evaluate.compute_metrics(y, y)["qwk"] == 1.0
    assert far < near


def test_nbsvm_learns_separable_toy_corpus():
    from sklearn.feature_extraction.text import TfidfVectorizer
    X, y = toy_corpus()
    A = TfidfVectorizer().fit_transform(X)
    clf = NBSVM(C=4).fit(A[::2], y[::2])
    assert (clf.predict(A[1::2]) == y[1::2]).mean() > 0.9
    P = clf.predict_proba(A[1::2])
    assert P.shape == (len(y[1::2]), 4) and np.allclose(P.sum(axis=1), 1)


def test_severe_multiplier_reaches_target():
    y = np.array([3, 3, 3, 3, 1])
    P = np.array([[.1, .5, 0, .4]] * 4 + [[.1, .8, 0, .1]])
    s = extensions.severe_multiplier(P, y, target=0.75)
    assert np.isclose(s, 1.3)  # 1.25 only ties Mild (0.5 vs 0.5) and argmax keeps the first class
    assert extensions.severe_multiplier(np.eye(4)[[3, 3]], [3, 3]) == 1.0


def test_find_label_issues_flags_flipped_labels(monkeypatch):
    monkeypatch.setitem(config.VECTORIZER_PARAMS, "min_df", 1)
    X, y = toy_corpus(40)
    noisy = y.copy()
    flipped = np.arange(0, len(y), 20)  # 8 posts moved to a far class
    noisy[flipped] = 3 - noisy[flipped]
    flagged, conf = extensions.find_label_issues(X, noisy, n_jobs=1)
    assert flagged[flipped].mean() >= 0.75
    assert flagged.mean() < 0.15
    assert conf.shape == y.shape


@pytest.mark.parametrize("model", list(config.MODELS))
@pytest.mark.parametrize("weighted", [False, True])
def test_dsas_pipeline_fits(model, weighted, monkeypatch):
    monkeypatch.setitem(config.VECTORIZER_PARAMS, "min_df", 1)
    X, y = toy_corpus()
    pipe = extensions.dsas_pipeline(model, weighted)
    if model == "rf":
        pipe.set_params(clf__n_estimators=10)
    assert (pipe.fit(X, y).predict(X) == y).mean() > 0.8


def test_selected_pipeline_uses_grid_choice():
    nb = extensions.selected_pipeline("tfidf", "nb")
    lr = extensions.selected_pipeline("bow", "lr")
    assert type(nb.named_steps["clf"]).__name__ == "ComplementNB" and nb.named_steps["clf"].alpha == 0.5
    assert lr.named_steps["clf"].l1_ratio == 1.0 and "scale" in lr.named_steps
    # SELECTED itself must not be mutated by fitting a pipeline built from it
    assert extensions.selected_pipeline("tfidf", "nb").named_steps["clf"] is not nb.named_steps["clf"]


def test_explain_returns_one_column_per_symptom(monkeypatch):
    monkeypatch.setitem(config.VECTORIZER_PARAMS, "min_df", 1)
    X, y = toy_corpus()
    lr = extensions.selected_pipeline("tfidf", "lr").fit(X, y)
    out = lexicon.explain(lr, ["hopeless worthless die"])
    assert list(out.columns) == ["predicted", *lexicon.SYMPTOMS]
    assert out.loc[0, "predicted"] == "Severe"
    assert out.loc[0, "Suicidal thoughts"] > 0
