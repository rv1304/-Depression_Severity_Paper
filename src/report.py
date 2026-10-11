"""Merge per-run results into tables, figures and LaTeX snippets for the paper.

Usage:
    python -m src.report                 # after any subset of funnel runs has finished

Reads   results/runs/*.json, results/confusion_*.csv, results/models/*.joblib,
        results_with_duplicates/runs/*.json (sensitivity run, optional)
Writes  results/metrics.csv, results/classification_reports.txt, results/figures/*,
        paper/generated/*.tex (\\input by paper/main.tex)
"""
import json
import logging
import os
import platform
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config, evaluate, plots

log = logging.getLogger("report")

SENS_DIR = config.ROOT / "results_with_duplicates"
GEN_DIR = config.ROOT / "paper" / "generated"
SHORT_EXP = {"bow": "BoW", "tfidf": "TF-IDF"}


def load_runs(out: Path) -> pd.DataFrame:
    files = sorted((out / "runs").glob("*.json")) if (out / "runs").exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(f.read_text()) for f in files])
    order = {k: i for i, k in enumerate(config.EXPERIMENTS)}, {k: i for i, k in enumerate(config.MODELS)}
    df["_e"], df["_m"] = df["experiment"].map(order[0]), df["model"].map(order[1])
    return df.sort_values(["_e", "_m"]).drop(columns=["_e", "_m"]).reset_index(drop=True)


def format_params(model: str, best_params: str) -> str:
    """Compact, LaTeX-safe description of a selected configuration."""
    p = json.loads(best_params)
    if model == "nb":
        variant = "Complement" if "Complement" in p.get("clf", "") else "Multinomial"
        return rf"{variant}, $\alpha$={float(p['clf__alpha']):g}"
    if model == "lr":
        return rf"$C$={float(p['clf__C']):g}, $\rho$={float(p['clf__l1_ratio']):g}"
    if model == "rf":
        return (f"{p['clf__n_estimators']} trees, depth {p['clf__max_depth']}, "
                f"split {p['clf__min_samples_split']}")
    return ""


def _bold_max(values: pd.Series, fmt: str = "{:.3f}") -> list:
    best = values.max()
    return [rf"\textbf{{{fmt.format(v)}}}" if np.isclose(v, best) else fmt.format(v) for v in values]


def table_metrics(m: pd.DataFrame) -> str:
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "severe_recall"]
    cells = {c: _bold_max(m[c]) for c in cols}
    lines = [r"\begin{tabular}{llrrrrrrl}", r"\toprule",
             r"Features & Model & Acc. & Macro-P & Macro-R & Macro-F1 & Severe-R & CV F1 ($\pm$sd) & Selected configuration \\",
             r"\midrule"]
    for i, r in m.iterrows():
        lines.append(" & ".join([
            SHORT_EXP[r.experiment], config.MODELS[r.model], *[cells[c][i] for c in cols],
            f"{r.cv_macro_f1_mean:.3f} $\\pm$ {r.cv_macro_f1_std:.3f}",
            format_params(r.model, r.best_params),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_timing(m: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Features & Model & Train (s) & Predict (ms / 1k posts) \\", r"\midrule"]
    for _, r in m.iterrows():
        lines.append(f"{SHORT_EXP[r.experiment]} & {config.MODELS[r.model]} & "
                     f"{r.fit_seconds:.1f} & {r.predict_ms_per_1k:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_leakage(m: pd.DataFrame, sens: pd.DataFrame) -> str:
    merged = m.merge(sens, on=["experiment", "model"], suffixes=("", "_dup"))
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Features & Model & Audited & As distributed & $\Delta$ (points) \\", r"\midrule"]
    for _, r in merged.iterrows():
        delta = 100 * (r.macro_f1_dup - r.macro_f1)
        lines.append(f"{SHORT_EXP[r.experiment]} & {config.MODELS[r.model]} & {r.macro_f1:.3f} & "
                     f"{r.macro_f1_dup:.3f} & {delta:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), merged


def hardware() -> str:
    ram = ""
    try:
        ram = f", {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.0f} GB RAM"
    except (ValueError, OSError, AttributeError):
        pass
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU cores{ram}"


def write_gen(name: str, text: str, gen_dir: Path) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{name}.tex").write_text(text + "%\n")


def build(out: Path = config.RESULTS_DIR, sens_out: Path = SENS_DIR, gen_dir: Path = GEN_DIR) -> pd.DataFrame:
    m = load_runs(out)
    if m.empty:
        log.warning("No finished runs in %s/runs yet", out)
        return m
    fig_dir = out / "figures"
    m.to_csv(out / "metrics.csv", index=False)
    reports = sorted((out / "runs").glob("*_report.txt"))
    (out / "classification_reports.txt").write_text("\n\n".join(f.read_text() for f in reports))

    plots.grouped_metric(m, "macro_f1", "Macro F1 (test)", "fig1_macro_f1", fig_dir)
    plots.grouped_metric(m, "severe_recall", "Severe-class recall (test)", "fig1b_severe_recall", fig_dir)
    plots.timing(m, fig_dir)
    matrices = {(r.model, r.experiment): np.loadtxt(out / f"confusion_{r.model}_{r.experiment}.csv", delimiter=",")
                for r in m.itertuples() if (out / f"confusion_{r.model}_{r.experiment}.csv").exists()}
    if matrices:
        plots.confusion_grid(matrices, fig_dir)

    stats_path = out / "corpus_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    lr_path, rf_path = out / "models" / "lr_tfidf.joblib", out / "models" / "rf_tfidf.joblib"
    if lr_path.exists():
        lr = joblib.load(lr_path)
        coefs = evaluate.lr_top_coefficients(lr)
        coefs.to_csv(out / "lr_top_coefficients_tfidf.csv", index=False)
        stats["lr_zero_weight_share"] = evaluate.lr_sparsity(lr)
        plots.signed_coefficients(coefs, "LR top coefficients, Severe class (TF-IDF)", "fig3_lr_coefficients", fig_dir)
        write_gen("lr_sparsity", f"{100 * stats['lr_zero_weight_share']:.1f}\\%", gen_dir)
    if rf_path.exists():
        imp = evaluate.rf_top_importances(joblib.load(rf_path))
        imp.to_csv(out / "rf_top_importances_tfidf.csv", index=False)
        plots.importances(imp, "RF top-20 Gini importances (TF-IDF)", "fig4_rf_importances", fig_dir)
    if stats:
        stats_path.write_text(json.dumps(stats, indent=2))

    # LaTeX snippets + figures, so paper/ compiles on its own (e.g. uploaded to Overleaf)
    paper_fig = gen_dir.parent / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    for pdf in fig_dir.glob("*.pdf"):
        shutil.copy2(pdf, paper_fig / pdf.name)
    write_gen("table_metrics", table_metrics(m), gen_dir)
    write_gen("table_timing", table_timing(m), gen_dir)
    write_gen("hardware", hardware(), gen_dir)
    write_gen("sklearn", sklearn.__version__, gen_dir)
    write_gen("lr_epochs", str(config.LR_MAX_ITER), gen_dir)
    best = m.loc[m["macro_f1"].idxmax()]
    write_gen("abs_best", f"{config.MODELS[best.model]} with {SHORT_EXP[best.experiment]} features achieved "
                          f"the highest macro-averaged F1-score of {best.macro_f1:.3f} "
                          f"(accuracy {best.accuracy:.3f})", gen_dir)
    sev = m.loc[m["severe_recall"].idxmax()]
    write_gen("abs_severe", f"{sev.severe_recall:.3f} ({config.MODELS[sev.model]}, {SHORT_EXP[sev.experiment]})",
              gen_dir)
    sens = load_runs(sens_out)
    if not sens.empty:
        tex, merged = table_leakage(m, sens)
        write_gen("table_leakage", tex, gen_dir)
        write_gen("abs_leak", f"{100 * (merged.macro_f1_dup - merged.macro_f1).max():.1f}", gen_dir)

    cols = ["experiment", "model", "accuracy", "macro_f1", "severe_recall", "cv_macro_f1_mean", "fit_seconds"]
    log.info("Merged %d runs\n%s", len(m), m[cols].round(4).to_string(index=False))
    return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build()
