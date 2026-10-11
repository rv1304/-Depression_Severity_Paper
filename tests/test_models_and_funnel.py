import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid

from src import config, evaluate
from src.funnel import run_experiment
from src.models import build_pipeline

WORDS = {
    0: "fine happy good day work friend",
    1: "tired bit sad stress work",
    2: "sad lonely not sleep cry",
    3: "hopeless worthless die end pain alone",
}


def toy_corpus(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, vocab in WORDS.items():
        words = vocab.split()
        for _ in range(n_per_class):
            X.append(" ".join(rng.choice(words, size=8)))
            y.append(label)
    return np.array(X), np.array(y)


@pytest.mark.parametrize("exp", list(config.EXPERIMENTS))
@pytest.mark.parametrize("model", list(config.MODELS))
def test_every_grid_param_is_valid_for_pipeline(exp, model):
    pipe, grid = build_pipeline(exp, model)
    for params in ParameterGrid(grid):
        pipe.set_params(**params)  # raises on a misspelled key


def test_rf_grid_matches_blueprint():
    _, grid = build_pipeline("tfidf", "rf")
    assert len(ParameterGrid(grid)) == 27


def test_lr_bow_pipeline_scales_counts_but_tfidf_does_not():
    bow, _ = build_pipeline("bow", "lr")
    tfidf, _ = build_pipeline("tfidf", "lr")
    assert list(bow.named_steps) == ["vec", "scale", "clf"]
    assert list(tfidf.named_steps) == ["vec", "clf"]
    assert list(build_pipeline("bow", "rf")[0].named_steps) == ["vec", "clf"]


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        build_pipeline("w2v", "rf")
    with pytest.raises(ValueError):
        build_pipeline("bow", "svm")


def test_compute_metrics_perfect_and_severe_recall():
    y = np.array([0, 1, 2, 3, 3])
    m = evaluate.compute_metrics(y, y)
    assert m["macro_f1"] == 1.0 and m["severe_recall"] == 1.0
    m = evaluate.compute_metrics(y, np.array([0, 1, 2, 2, 3]))
    assert m["severe_recall"] == 0.5


def test_normalized_confusion_rows_sum_to_one():
    y = np.array([0, 1, 2, 3, 3, 2])
    cm = evaluate.normalized_confusion(y, np.array([0, 2, 2, 3, 1, 2]))
    assert cm.shape == (4, 4)
    assert np.allclose(cm.sum(axis=1), 1.0)


@pytest.mark.parametrize("model", list(config.MODELS))
def test_run_experiment_smoke(model, monkeypatch):
    monkeypatch.setitem(config.VECTORIZER_PARAMS, "min_df", 1)
    X, y = toy_corpus()
    # Full grid for the cheap models (best config is then well-defined); one random config for RF.
    search = "random" if model == "rf" else "grid"
    row, searcher, best, y_pred = run_experiment(
        "tfidf", model, X[::2], y[::2], X[1::2], y[1::2],
        search=search, n_iter=1, cv_folds=2, n_jobs=1,
    )
    assert len(y_pred) == len(X[1::2])
    assert row["macro_f1"] > 0.5  # toy classes are separable
    assert row["n_configs"] == (1 if model == "rf" else 6)
    if model == "lr":
        assert not evaluate.lr_top_coefficients(best, k=3).empty
    if model == "rf":
        assert len(evaluate.rf_top_importances(best, k=3)) == 3
