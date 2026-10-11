"""End-to-end funnel: data -> preprocessing -> {NB, LR, RF} x {BoW, TF-IDF} -> metrics, tables, figures.

Usage:
    python -m src.funnel                         # full blueprint grid, all models, both experiments
    python -m src.funnel --models rf --search random --n-iter 8
    python -m src.funnel --sample 3000           # quick smoke run on a stratified subsample
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split

from src import config, evaluate, plots
from src.data import ensure_dataset, prepare
from src.models import build_pipeline
from src.preprocess import clean_series

log = logging.getLogger("funnel")


def load_clean(refresh: bool = False, dedupe: bool = True) -> tuple[pd.DataFrame, dict]:
    """Prepared + preprocessed corpus, cached to data/ (lemmatisation is the slow step)."""
    suffix = "" if dedupe else "_with_duplicates"
    clean_csv = config.DATA_DIR / f"clean{suffix}.csv"
    stats_path = config.DATA_DIR / f"clean{suffix}_stats.json"
    if clean_csv.exists() and stats_path.exists() and not refresh:
        df = pd.read_csv(clean_csv, keep_default_na=False)
        return df, json.loads(stats_path.read_text())
    raw = pd.read_csv(ensure_dataset())
    df, stats = prepare(raw, dedupe=dedupe)
    log.info("Preprocessing %d texts", len(df))
    df["clean"] = clean_series(df["text"])
    empty = df["clean"] == ""
    stats["dropped_empty_after_cleaning"] = int(empty.sum())
    df = df[~empty].reset_index(drop=True)
    stats["final_rows"] = len(df)
    stats["class_counts"] = {
        config.CLASS_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()
    }
    stats["mean_tokens"] = float(df["clean"].str.split().str.len().mean())
    df.to_csv(clean_csv, index=False)
    stats_path.write_text(json.dumps(stats, indent=2))
    return df, stats


def run_experiment(experiment, model, X_train, y_train, X_test, y_test,
                   search="grid", n_iter=10, cv_folds=config.CV_FOLDS, n_jobs=-1, seed=config.SEED):
    """Tune with stratified k-fold CV on train, refit best, evaluate once on held-out test."""
    pipe, grid = build_pipeline(experiment, model, seed)
    if model == "rf" and (n_jobs < 0 or n_jobs > config.RF_MAX_PARALLEL_FITS):
        n_jobs = config.RF_MAX_PARALLEL_FITS
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    common = dict(scoring="f1_macro", cv=cv, n_jobs=n_jobs, refit=True, return_train_score=True)
    if search == "random":
        searcher = RandomizedSearchCV(pipe, grid, n_iter=n_iter, random_state=seed, **common)
    else:
        searcher = GridSearchCV(pipe, grid, **common)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        searcher.fit(X_train, y_train)
        search_seconds = time.perf_counter() - t0
    n_convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    best = searcher.best_estimator_
    # Re-time a single fit of the chosen config so efficiency numbers are comparable across models.
    t0 = time.perf_counter()
    best.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = best.predict(X_test)
    predict_seconds = time.perf_counter() - t0

    i = searcher.best_index_
    row = {
        "experiment": experiment,
        "model": model,
        **evaluate.compute_metrics(y_test, y_pred),
        "cv_macro_f1_mean": searcher.cv_results_["mean_test_score"][i],
        "cv_macro_f1_std": searcher.cv_results_["std_test_score"][i],
        "cv_train_macro_f1_mean": searcher.cv_results_["mean_train_score"][i],
        "best_params": json.dumps({k: str(v) for k, v in searcher.best_params_.items()}),
        "n_configs": len(searcher.cv_results_["params"]),
        "search_seconds": search_seconds,
        "fit_seconds": fit_seconds,
        "predict_ms_per_1k": 1000 * predict_seconds / max(len(y_test), 1) * 1000,
        "vocab_size": len(best.named_steps["vec"].vocabulary_),
        "convergence_warnings": n_convergence,
    }
    return row, searcher, best, y_pred


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(config.MODELS), choices=list(config.MODELS))
    ap.add_argument("--experiments", nargs="+", default=list(config.EXPERIMENTS),
                    choices=list(config.EXPERIMENTS))
    ap.add_argument("--search", choices=["grid", "random"], default="grid")
    ap.add_argument("--n-iter", type=int, default=10, help="configs per model for --search random")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--sample", type=int, default=None, help="stratified subsample size (smoke runs)")
    ap.add_argument("--refresh", action="store_true", help="redo preprocessing, ignore cache")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="sensitivity run on the corpus as distributed (no de-duplication)")
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = args.out
    fig_dir, model_dir = out / "figures", out / "models"
    for d in (out, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    df, stats = load_clean(args.refresh, dedupe=not args.keep_duplicates)
    if args.sample and args.sample < len(df):
        df, _ = train_test_split(df, train_size=args.sample, stratify=df["label"],
                                 random_state=config.SEED)
        stats["sampled_rows"] = len(df)
    log.info("Corpus: %s", json.dumps(stats))

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].values, df["label"].values, test_size=config.TEST_SIZE,
        stratify=df["label"].values, random_state=config.SEED,
    )
    stats.update(train_rows=len(X_train), test_rows=len(X_test))
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    plots.class_distribution(stats["class_counts"], fig_dir)

    # Each model x experiment writes its own files under runs/, so models can run as separate
    # processes (e.g. `--models rf` and `--models lr` in parallel); src.report merges whatever exists.
    run_dir = out / "runs"
    run_dir.mkdir(exist_ok=True)
    for exp in args.experiments:
        for model in args.models:
            log.info("Running %s | %s (%s search)", config.MODELS[model], config.EXPERIMENTS[exp], args.search)
            row, searcher, best, y_pred = run_experiment(
                exp, model, X_train, y_train, X_test, y_test,
                search=args.search, n_iter=args.n_iter, n_jobs=args.n_jobs,
            )
            log.info("  macro-F1=%.4f  severe-recall=%.4f  best=%s",
                     row["macro_f1"], row["severe_recall"], row["best_params"])
            tag = f"{model}_{exp}"
            row["search"] = args.search
            (run_dir / f"{tag}.json").write_text(json.dumps(row, indent=2, default=float))
            pd.DataFrame(searcher.cv_results_).to_csv(out / f"cv_results_{tag}.csv", index=False)
            np.savetxt(out / f"confusion_{tag}.csv", evaluate.normalized_confusion(y_test, y_pred),
                       delimiter=",", fmt="%.4f")
            (run_dir / f"{tag}_report.txt").write_text(
                f"=== {config.MODELS[model]} | {config.EXPERIMENTS[exp]} ===\n{evaluate.report(y_test, y_pred)}")
            joblib.dump(best, model_dir / f"{tag}.joblib")

    from src import report  # local import: report depends on plots/matplotlib
    return report.build(out)


if __name__ == "__main__":
    main()
