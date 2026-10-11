import json

import numpy as np

from src import report


def _row(model, exp, f1, severe):
    params = {
        "nb": {"clf": "ComplementNB()", "clf__alpha": "0.5"},
        "lr": {"clf__C": "10.0", "clf__l1_ratio": "1.0"},
    }[model]
    return {
        "experiment": exp, "model": model, "accuracy": f1 + 0.01, "macro_precision": f1,
        "macro_recall": f1, "macro_f1": f1, "severe_recall": severe, "cv_macro_f1_mean": f1,
        "cv_macro_f1_std": 0.01, "best_params": json.dumps(params), "fit_seconds": 1.5,
        "predict_ms_per_1k": 2.0,
    }


def _write_runs(out, rows):
    (out / "runs").mkdir(parents=True)
    for r in rows:
        tag = f"{r['model']}_{r['experiment']}"
        (out / "runs" / f"{tag}.json").write_text(json.dumps(r))
        (out / "runs" / f"{tag}_report.txt").write_text(tag)
        np.savetxt(out / f"confusion_{tag}.csv", np.eye(4), delimiter=",")


def test_format_params():
    assert report.format_params("nb", json.dumps({"clf": "ComplementNB()", "clf__alpha": "0.5"})) \
        == r"Complement, $\alpha$=0.5"
    assert report.format_params("rf", json.dumps(
        {"clf__n_estimators": 500, "clf__max_depth": 50, "clf__min_samples_split": 2})) \
        == "500 trees, depth 50, split 2"


def test_build_with_no_runs_returns_empty(tmp_path):
    assert report.build(tmp_path, tmp_path / "none", tmp_path / "gen").empty


def test_build_merges_runs_and_writes_latex(tmp_path):
    out, sens, gen = tmp_path / "results", tmp_path / "dup", tmp_path / "gen"
    _write_runs(out, [_row("lr", "tfidf", 0.70, 0.80), _row("nb", "bow", 0.65, 0.70)])
    _write_runs(sens, [_row("nb", "bow", 0.75, 0.72)])

    m = report.build(out, sens, gen)

    assert list(zip(m.experiment, m.model)) == [("bow", "nb"), ("tfidf", "lr")]  # config order
    assert (out / "metrics.csv").exists()
    assert (out / "figures" / "fig1_macro_f1.pdf").exists()
    assert (out / "figures" / "fig2_confusion_matrices.pdf").exists()
    metrics_tex = (gen / "table_metrics.tex").read_text()
    assert r"\textbf{0.700}" in metrics_tex  # best macro-F1 bolded
    assert "Logistic Regression with TF-IDF" in (gen / "abs_best.tex").read_text()
    assert (gen / "abs_leak.tex").read_text().startswith("10.0")  # 0.75 - 0.65
    assert "+10.0" in (gen / "table_leakage.tex").read_text()
