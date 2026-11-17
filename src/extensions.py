"""Extended experiments added in revision (E1-E8). Each part writes results/extensions/<part>.json;
`report` turns whatever exists into tables, figures and LaTeX snippets (paper/generated/ext_*.tex).

Parts:
    ablation  E1 Exp C (TF-IDF + DSAS lexicon), E2 training-set label cleaning, E3 NB-SVM,
              E4 Severe-threshold tuning -- all on the original seed-42 split and test set
    leakage   E5 LR and RF on the corpus as distributed (NB is already in results_with_duplicates/)
    seeds     E6 the selected configurations re-run on 5 stratified splits
    symptoms  E7 DSAS symptom profile per severity class + symptom-level explanations
    noise     E8 confident-learning label-noise filtering of the whole audited corpus
    report    tables, figures, LaTeX

Usage:
    python -m src.extensions --parts ablation leakage noise symptoms
    python -m src.extensions --parts seeds --n-jobs 3
    python -m src.extensions --parts report

Expensive grid searches are not repeated: every model uses the configuration the original grid search
selected (results/metrics.csv), so new rows differ from the published ones in exactly one factor.
"""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config, evaluate, lexicon, plots
from src.funnel import load_clean
from src.models import NBSVM, build_pipeline

log = logging.getLogger("extensions")

EXT_DIR = config.RESULTS_DIR / "extensions"
GEN_DIR = config.ROOT / "paper" / "generated"
SEEDS = [42, 43, 44, 45, 46]
SEVERE_TARGET = 0.80
NBSVM_GRID = {"clf__C": [1.0, 4.0, 10.0]}
SHORT_EXP = {"bow": "BoW", "tfidf": "TF-IDF", "dsas": "TF-IDF + DSAS", "dsas_w": "DSAS-weighted TF-IDF"}

# Configurations selected by the original exhaustive grid search (results/metrics.csv).
SELECTED = {
    ("bow", "nb"): {"clf": MultinomialNB(alpha=0.5)},
    ("tfidf", "nb"): {"clf": ComplementNB(alpha=0.5)},
    ("bow", "lr"): {"clf__C": 1.0, "clf__l1_ratio": 1.0},
    ("tfidf", "lr"): {"clf__C": 1.0, "clf__l1_ratio": 0.5},
    ("bow", "rf"): {"clf__n_estimators": 500, "clf__max_depth": 100, "clf__min_samples_split": 10},
    ("tfidf", "rf"): {"clf__n_estimators": 500, "clf__max_depth": 100, "clf__min_samples_split": 10},
}


# ---------------------------------------------------------------------------- pipelines
def selected_pipeline(experiment: str, model: str, rf_jobs: int = 1) -> Pipeline:
    """Original BoW / TF-IDF pipeline with its grid-selected configuration."""
    pipe, _ = build_pipeline(experiment, model)
    pipe.set_params(**clone(SELECTED[(experiment, model)], safe=False))
    if model == "rf":
        pipe.set_params(clf__n_jobs=rf_jobs)
    return pipe


def dsas_pipeline(model: str, weighted: bool = False, rf_jobs: int = 1) -> Pipeline:
    """Experiment C. weighted=False: TF-IDF + 11 DSAS symptom features (seed lexicon).
    weighted=True: DSAS-weighted TF-IDF (expanded lexicon, weight 2) + expanded symptom features.
    The classifier keeps the configuration selected for plain TF-IDF."""
    if weighted:
        feats = FeatureUnion([("tfidf", lexicon.LexiconWeightedTfidf("expanded", 2.0)),
                              ("sym", lexicon.SymptomFeatures("expanded"))])
    else:
        feats = FeatureUnion([("tfidf", TfidfVectorizer(**config.VECTORIZER_PARAMS)),
                              ("sym", lexicon.SymptomFeatures("seed"))])
    clf = selected_pipeline("tfidf", model, rf_jobs).named_steps["clf"]
    return Pipeline([("feats", feats), ("clf", clf)])


def nbsvm_pipeline(C: float = 4.0) -> Pipeline:
    return Pipeline([("vec", TfidfVectorizer(**config.VECTORIZER_PARAMS)), ("clf", NBSVM(C=C))])


# ---------------------------------------------------------------------------- helpers
def split(df: pd.DataFrame, seed: int = config.SEED):
    return train_test_split(df["clean"].values, df["label"].values, test_size=config.TEST_SIZE,
                            stratify=df["label"].values, random_state=seed)


def fit_eval(pipe, X_train, y_train, X_test, y_test, **info) -> tuple:
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    y_pred = pipe.predict(X_test)
    row = {**info, **evaluate.compute_metrics(y_test, y_pred), "fit_seconds": fit_seconds,
           "train_rows": len(y_train), "test_rows": len(y_test)}
    log.info("  %s  macro-F1=%.4f  severe-R=%.4f  QWK=%.4f  (%.0fs)", info, row["macro_f1"],
             row["severe_recall"], row["qwk"], fit_seconds)
    return row, pipe, y_pred


def find_label_issues(texts, y, seed: int = config.SEED, n_jobs: int = -1) -> tuple:
    """Confident learning (Northcutt et al. 2021), simplified: out-of-fold probabilities averaged over a
    TF-IDF Logistic Regression and a Complement NB. A post is flagged when another class is predicted
    with at least that class's average self-confidence and its own label falls below its class's
    average self-confidence. Vectorisers are fitted inside each fold."""
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    vec = lambda: TfidfVectorizer(**config.VECTORIZER_PARAMS)  # noqa: E731
    probes = [Pipeline([("vec", vec()), ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=1000))]),
              Pipeline([("vec", vec()), ("clf", ComplementNB(alpha=0.5))])]
    P = np.mean([cross_val_predict(p, texts, y, cv=cv, method="predict_proba", n_jobs=n_jobs) for p in probes],
                axis=0)
    idx = np.arange(len(y))
    self_conf = P[idx, y]
    thresholds = np.array([self_conf[y == k].mean() for k in range(len(config.CLASS_NAMES))])
    pred = P.argmax(axis=1)
    flagged = (pred != y) & (P[idx, pred] >= thresholds[pred]) & (self_conf < thresholds[y])
    return flagged, self_conf


def severe_multiplier(P: np.ndarray, y, target: float = SEVERE_TARGET) -> float:
    """Smallest factor on the Severe probability whose out-of-fold Severe recall reaches `target`."""
    y = np.asarray(y)
    for s in np.round(np.arange(1.0, 6.01, 0.05), 2):
        Q = P.copy()
        Q[:, config.SEVERE] *= s
        if (Q.argmax(axis=1)[y == config.SEVERE] == config.SEVERE).mean() >= target:
            return float(s)
    return float(s)


def save(part: str, payload) -> None:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    (EXT_DIR / f"{part}.json").write_text(json.dumps(payload, indent=2, default=float))


def load(part: str):
    path = EXT_DIR / f"{part}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------- parts
def part_ablation(df, n_jobs):
    X_train, X_test, y_train, y_test = split(df)
    rows = []
    # E1: Experiment C, both lexicon variants, all three classifiers.
    for weighted, exp in ((False, "dsas"), (True, "dsas_w")):
        for model in config.MODELS:
            row, _, _ = fit_eval(dsas_pipeline(model, weighted, rf_jobs=n_jobs), X_train, y_train, X_test, y_test,
                                 experiment=exp, model=model, variant="exp_c")
            rows.append(row)
    # E3: NB-SVM, C chosen by stratified 5-fold CV on the training set only.
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(nbsvm_pipeline(), NBSVM_GRID, scoring="f1_macro", cv=cv, n_jobs=n_jobs).fit(X_train, y_train)
    nbsvm_c = float(search.best_params_["clf__C"])
    row, _, _ = fit_eval(nbsvm_pipeline(nbsvm_c), X_train, y_train, X_test, y_test,
                         experiment="tfidf", model="nbsvm", variant="nbsvm")
    row["cv_macro_f1_mean"] = float(search.best_score_)
    rows.append(row)
    # E2: drop likely label errors from the TRAINING set only; the test set is untouched.
    flagged, _ = find_label_issues(X_train, y_train, n_jobs=n_jobs)
    keep = ~flagged
    cleaning = {"train_flagged": int(flagged.sum()), "train_flagged_pct": 100 * float(flagged.mean()),
                "flagged_per_class": dict(zip(config.CLASS_NAMES, np.bincount(y_train[flagged], minlength=4).tolist()))}
    log.info("Training-set label issues: %s", cleaning)
    candidates = {("tfidf", "lr"): selected_pipeline("tfidf", "lr"), ("bow", "nb"): selected_pipeline("bow", "nb"),
                  ("tfidf", "nbsvm"): nbsvm_pipeline(nbsvm_c)}
    for (exp, model), pipe in candidates.items():
        row, _, _ = fit_eval(pipe, X_train[keep], y_train[keep], X_test, y_test,
                             experiment=exp, model=model, variant="train_cleaned")
        rows.append(row)
    # E4: Severe threshold chosen on out-of-fold training probabilities, applied once to the test set.
    for (exp, model), pipe in {("bow", "nb"): selected_pipeline("bow", "nb"),
                               ("tfidf", "lr"): selected_pipeline("tfidf", "lr"),
                               ("tfidf", "nbsvm"): nbsvm_pipeline(nbsvm_c)}.items():
        P_oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)
        s = severe_multiplier(P_oof, y_train)
        P_test = pipe.fit(X_train, y_train).predict_proba(X_test)
        P_test[:, config.SEVERE] *= s
        y_pred = P_test.argmax(axis=1)
        m = evaluate.compute_metrics(y_test, y_pred)
        sev_precision = float((y_test[y_pred == config.SEVERE] == config.SEVERE).mean())
        rows.append({"experiment": exp, "model": model, "variant": "severe_threshold", **m,
                     "severe_multiplier": s, "severe_precision": sev_precision})
        log.info("  threshold %s/%s s=%.2f severe-R=%.4f severe-P=%.4f macro-F1=%.4f",
                 exp, model, s, m["severe_recall"], sev_precision, m["macro_f1"])
    save("ablation", {"nbsvm_C": nbsvm_c, "train_cleaning": cleaning, "rows": rows})


def part_leakage(n_jobs):
    dirty, stats = load_clean(dedupe=False)
    X_train, X_test, y_train, y_test = split(dirty)
    # Share of test posts whose exact text also occurs in the training set.
    overlap = float(np.isin(X_test, X_train).mean())
    rows = []
    for exp in config.EXPERIMENTS:
        for model in ("lr", "rf"):
            row, _, _ = fit_eval(selected_pipeline(exp, model, rf_jobs=n_jobs), X_train, y_train, X_test, y_test,
                                 experiment=exp, model=model, variant="as_distributed")
            rows.append(row)
    save("leakage", {"test_text_in_train_pct": 100 * overlap, "rows_dirty": len(dirty), "rows": rows})


def _seed_run(df, seed, exp, model, nbsvm_c):
    X_train, X_test, y_train, y_test = split(df, seed)
    pipe = nbsvm_pipeline(nbsvm_c) if model == "nbsvm" else selected_pipeline(exp, model)
    row, _, _ = fit_eval(pipe, X_train, y_train, X_test, y_test, experiment=exp, model=model, seed=seed)
    return row


def part_seeds(df, n_jobs):
    abl = load("ablation")
    nbsvm_c = abl["nbsvm_C"] if abl else 4.0
    jobs = [(seed, exp, model) for seed in SEEDS for exp in config.EXPERIMENTS for model in config.MODELS]
    jobs += [(seed, "tfidf", "nbsvm") for seed in SEEDS]
    rows = Parallel(n_jobs=n_jobs)(delayed(_seed_run)(df, s, e, m, nbsvm_c) for s, e, m in jobs)
    save("seeds", {"seeds": SEEDS, "nbsvm_C": nbsvm_c, "rows": rows})


def part_symptoms(df):
    profile = lexicon.symptom_profile(df["clean"], df["label"])
    lex_sizes = {"seed": sum(map(len, lexicon.seed_lexicon().values())),
                 "expanded": sum(map(len, lexicon.expanded_lexicon().values()))}
    coverage = float((lexicon.symptom_counts(df["clean"], lexicon.seed_lexicon()).sum(axis=1) > 0).mean())
    # Explanations from the published TF-IDF Logistic Regression configuration on the seed-42 split.
    X_train, X_test, y_train, y_test = split(df)
    lr = selected_pipeline("tfidf", "lr").fit(X_train, y_train)
    expl = lexicon.explain(lr, X_test)
    mean_contrib = expl.groupby("predicted")[lexicon.SYMPTOMS].mean().reindex(config.CLASS_NAMES)
    sev = expl[expl["predicted"] == "Severe"]
    top = sev[lexicon.SYMPTOMS].idxmax(axis=1)[sev[lexicon.SYMPTOMS].max(axis=1) > 0]
    examples = [  # illustrative sentences written for the paper, not dataset posts
        "I cannot sleep anymore, I feel useless and hopeless and I just want to die",
        "Work has been stressful and I am tired, but I am managing",
        "I feel lonely and empty most days and nothing interests me anymore",
    ]
    from src.preprocess import clean_series
    ex = lexicon.explain(lr, clean_series(pd.Series(examples)).values)
    ex.insert(0, "text", examples)
    save("symptoms", {
        "lexicon_terms": lex_sizes, "posts_with_any_symptom_pct": 100 * coverage,
        "profile_pct": profile.round(2).to_dict(orient="index"),
        "mean_contribution": mean_contrib.round(4).to_dict(orient="index"),
        "severe_pred_with_positive_symptom_pct": 100 * len(top) / max(len(sev), 1),
        "severe_pred_top_symptom_pct": (100 * top.value_counts() / max(len(sev), 1)).round(2).to_dict(),
        "examples": ex.round(4).to_dict(orient="records"),
    })


def part_noise(df, n_jobs):
    """E8: flag likely label errors in the whole audited corpus (out-of-fold, so no post is judged by a model
    that saw it), remove them, and re-run the benchmark. The flagging models are linear (LR, CNB), so
    Random Forest results on the filtered corpus act as a check that gains are not circular."""
    X, y = df["clean"].values, df["label"].values
    flagged, self_conf = find_label_issues(X, y, n_jobs=n_jobs)
    filtered = df[~flagged].reset_index(drop=True)
    info = {"flagged": int(flagged.sum()), "flagged_pct": 100 * float(flagged.mean()),
            "flagged_per_class": dict(zip(config.CLASS_NAMES, np.bincount(y[flagged], minlength=4).tolist())),
            "class_counts_after": dict(zip(config.CLASS_NAMES, np.bincount(filtered["label"], minlength=4).tolist()))}
    log.info("Corpus label issues: %s", info)
    X_train, X_test, y_train, y_test = split(filtered)
    rows = []
    for exp in config.EXPERIMENTS:
        for model in config.MODELS:
            row, _, _ = fit_eval(selected_pipeline(exp, model, rf_jobs=n_jobs), X_train, y_train, X_test, y_test,
                                 experiment=exp, model=model, variant="noise_filtered")
            rows.append(row)
    abl = load("ablation")
    nbsvm_c = abl["nbsvm_C"] if abl else 4.0
    row, _, _ = fit_eval(nbsvm_pipeline(nbsvm_c), X_train, y_train, X_test, y_test,
                         experiment="tfidf", model="nbsvm", variant="noise_filtered")
    rows.append(row)
    # Sensitivity curve: remove the lowest-self-confidence k% instead of the flag rule.
    curve = []
    for pct in (0, 5, 10, 15, 20, 25):
        keep = self_conf > np.percentile(self_conf, pct) if pct else np.ones(len(y), bool)
        a, b, c, d = split(df[keep].reset_index(drop=True))
        r, _, _ = fit_eval(nbsvm_pipeline(nbsvm_c), a, c, b, d, experiment="tfidf", model="nbsvm", removed_pct=pct)
        curve.append({"removed_pct": pct, "macro_f1": r["macro_f1"], "severe_recall": r["severe_recall"],
                      "qwk": r["qwk"]})
    save("noise", {**info, "rows": rows, "curve": curve})


# ---------------------------------------------------------------------------- report
def _tex_table(header: str, cols: str, lines: list) -> str:
    return "\n".join([rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule",
                      *[ln + r" \\" for ln in lines], r"\bottomrule", r"\end{tabular}"])


def _name(model: str) -> str:
    return {"nbsvm": "NB-SVM", **config.MODELS}[model]


def _write(name: str, text: str) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / f"{name}.tex").write_text(text + "%\n")


def part_report():
    base = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    fig_dir = config.RESULTS_DIR / "figures"
    paper_fig = config.ROOT / "paper" / "figures"
    seeds, abl, leak, noise, sym = load("seeds"), load("ablation"), load("leakage"), load("noise"), load("symptoms")

    if seeds:
        # Published runs predate QWK; the seed-42 re-run uses the same split and configuration, so it supplies it.
        s42 = pd.DataFrame(seeds["rows"]).query("seed == @config.SEED")[["experiment", "model", "qwk"]]
        base = base.drop(columns="qwk", errors="ignore").merge(s42, on=["experiment", "model"], how="left")
        s = pd.DataFrame(seeds["rows"])
        agg = s.groupby(["experiment", "model"]).agg(
            f1_m=("macro_f1", "mean"), f1_s=("macro_f1", "std"), sev_m=("severe_recall", "mean"),
            sev_s=("severe_recall", "std"), qwk_m=("qwk", "mean"), qwk_s=("qwk", "std"), acc_m=("accuracy", "mean"))
        order = [(e, m) for e in config.EXPERIMENTS for m in config.MODELS] + [("tfidf", "nbsvm")]
        lines = [f"{SHORT_EXP[e]} & {_name(m)} & {r.acc_m:.3f} & {r.f1_m:.3f} $\\pm$ {r.f1_s:.3f} & "
                 f"{r.sev_m:.3f} $\\pm$ {r.sev_s:.3f} & {r.qwk_m:.3f} $\\pm$ {r.qwk_s:.3f}"
                 for (e, m), r in agg.reindex(order).dropna().iterrows()]
        _write("ext_table_seeds", _tex_table(
            r"Features & Model & Acc. & Macro-F1 & Severe-R & QWK", "llrrrr", lines))
        s42 = s[s["seed"] == config.SEED].set_index(["experiment", "model"])
        _write("ext_table_qwk", _tex_table(r"Features & Model & QWK (seed 42)", "llr",
               [f"{SHORT_EXP[e]} & {_name(m)} & {s42.loc[(e, m), 'qwk']:.3f}" for e, m in order if (e, m) in s42.index]))

    if abl:
        a = pd.DataFrame(abl["rows"])
        b = base.set_index(["experiment", "model"])
        expc = a[a["variant"] == "exp_c"]
        lines = [f"TF-IDF (published) & {_name(m)} & {b.loc[('tfidf', m), 'macro_f1']:.3f} & "
                 f"{b.loc[('tfidf', m), 'severe_recall']:.3f} & --" for m in config.MODELS]
        lines += [f"{SHORT_EXP[r.experiment]} & {_name(r.model)} & {r.macro_f1:.3f} & {r.severe_recall:.3f} & "
                  f"{100 * (r.macro_f1 - b.loc[('tfidf', r.model), 'macro_f1']):+.1f}" for r in expc.itertuples()]
        _write("ext_table_expc", _tex_table(
            r"Features & Model & Macro-F1 & Severe-R & $\Delta$F1 (pts)", "llrrr", lines))

        def get(variant, exp, model, col="macro_f1"):
            sub = a[(a["variant"] == variant) & (a["experiment"] == exp) & (a["model"] == model)]
            return float(sub[col].iloc[0]) if len(sub) else float("nan")
        lr0 = b.loc[("tfidf", "lr")]
        rows = [("LR, TF-IDF (published best)", lr0.macro_f1, lr0.severe_recall, lr0.get("qwk", np.nan)),
                ("+ DSAS symptom features (Exp C)", get("exp_c", "dsas", "lr"), get("exp_c", "dsas", "lr", "severe_recall"),
                 get("exp_c", "dsas", "lr", "qwk")),
                ("+ DSAS-weighted TF-IDF (Exp C, expanded)", get("exp_c", "dsas_w", "lr"),
                 get("exp_c", "dsas_w", "lr", "severe_recall"), get("exp_c", "dsas_w", "lr", "qwk")),
                ("+ training-set label cleaning", get("train_cleaned", "tfidf", "lr"),
                 get("train_cleaned", "tfidf", "lr", "severe_recall"), get("train_cleaned", "tfidf", "lr", "qwk")),
                ("NB-SVM, TF-IDF", get("nbsvm", "tfidf", "nbsvm"), get("nbsvm", "tfidf", "nbsvm", "severe_recall"),
                 get("nbsvm", "tfidf", "nbsvm", "qwk")),
                ("NB-SVM + training-set label cleaning", get("train_cleaned", "tfidf", "nbsvm"),
                 get("train_cleaned", "tfidf", "nbsvm", "severe_recall"), get("train_cleaned", "tfidf", "nbsvm", "qwk"))]
        fmt = lambda v: "--" if np.isnan(v) else f"{v:.3f}"  # noqa: E731
        _write("ext_table_ablation", _tex_table(r"Setup & Macro-F1 & Severe-R & QWK", "lrrr",
               [f"{n} & {fmt(f)} & {fmt(r)} & {fmt(q)}" for n, f, r, q in rows]))
        thr = a[a["variant"] == "severe_threshold"]
        lines = []
        for r in thr.itertuples():
            ref = (b.loc[(r.experiment, r.model)] if (r.experiment, r.model) in b.index
                   else a[(a["variant"] == "nbsvm")].iloc[0])
            lines.append(f"{SHORT_EXP[r.experiment]} & {_name(r.model)} & {ref['severe_recall']:.3f} & "
                         f"\\textbf{{{r.severe_recall:.3f}}} & {r.severe_precision:.3f} & {ref['macro_f1']:.3f} & "
                         f"{r.macro_f1:.3f}")
        _write("ext_table_threshold", _tex_table(
            r"Features & Model & Sev-R before & Sev-R after & Sev-P after & F1 before & F1 after", "llrrrrr", lines))
        _write("ext_train_flagged", f"{abl['train_cleaning']['train_flagged_pct']:.1f}\\%")

    if leak:
        L = pd.DataFrame(leak["rows"]).set_index(["experiment", "model"])
        b = base.set_index(["experiment", "model"])
        lines = [f"{SHORT_EXP[e]} & {_name(m)} & {b.loc[(e, m), 'macro_f1']:.3f} & {L.loc[(e, m), 'macro_f1']:.3f} & "
                 f"{100 * (L.loc[(e, m), 'macro_f1'] - b.loc[(e, m), 'macro_f1']):+.1f}"
                 for e in config.EXPERIMENTS for m in ("lr", "rf")]
        _write("ext_table_leakage", "\n".join(lines))
        _write("ext_leak_overlap", f"{leak['test_text_in_train_pct']:.1f}\\%")

    if noise:
        N = pd.DataFrame(noise["rows"])
        b = base.set_index(["experiment", "model"])
        lines = []
        for r in N.itertuples():
            ref = b.loc[(r.experiment, r.model), "macro_f1"] if (r.experiment, r.model) in b.index else (
                pd.DataFrame(abl["rows"]).query("variant == 'nbsvm'").macro_f1.iloc[0] if abl else np.nan)
            lines.append(f"{SHORT_EXP[r.experiment]} & {_name(r.model)} & {ref:.3f} & {r.macro_f1:.3f} & "
                         f"{r.severe_recall:.3f} & {r.qwk:.3f}")
        _write("ext_table_noise", _tex_table(
            r"Features & Model & F1 audited & F1 filtered & Severe-R & QWK", "llrrrr", lines))
        _write("ext_noise_flagged", f"{noise['flagged_pct']:.1f}\\%")
        plots.noise_curve(pd.DataFrame(noise["curve"]), fig_dir)

    if sym:
        prof = pd.DataFrame(sym["profile_pct"]).T.reindex(config.CLASS_NAMES)
        plots.symptom_heatmap(prof, "DSAS symptom mentions by severity class", "fig6_symptom_profile", fig_dir)
        contrib = pd.DataFrame(sym["mean_contribution"]).T.reindex(config.CLASS_NAMES)
        plots.symptom_heatmap(contrib, "Mean symptom contribution to the predicted class (LR, TF-IDF)",
                              "fig8_symptom_contribution", fig_dir, fmt=".2f", cbar_label="logit contribution")
        lines = []
        for e in sym["examples"]:
            top = sorted(((v, k) for k, v in e.items() if k in lexicon.SYMPTOMS and v > 0), reverse=True)[:3]
            why = ", ".join(f"{k} ({v:+.2f})" for v, k in top) or "no lexicon symptom"
            lines.append(rf"\textit{{``{e['text']}''}} & {e['predicted']} & {why}")
        _write("ext_table_examples", _tex_table(r"Example text & Prediction & Top symptoms (contribution)",
                                                "p{0.42\\linewidth}lp{0.3\\linewidth}", lines))

    paper_fig.mkdir(parents=True, exist_ok=True)
    for pdf in fig_dir.glob("fig[678]_*.pdf"):
        shutil.copy2(pdf, paper_fig / pdf.name)
    log.info("Report written to %s", GEN_DIR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True,
                    choices=["ablation", "leakage", "seeds", "symptoms", "noise", "report"])
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df, _ = load_clean()
    for part in args.parts:
        log.info("=== %s ===", part)
        if part == "ablation":
            part_ablation(df, args.n_jobs)
        elif part == "leakage":
            part_leakage(args.n_jobs)
        elif part == "seeds":
            part_seeds(df, args.n_jobs)
        elif part == "symptoms":
            part_symptoms(df)
        elif part == "noise":
            part_noise(df, args.n_jobs)
        else:
            part_report()


if __name__ == "__main__":
    main()
