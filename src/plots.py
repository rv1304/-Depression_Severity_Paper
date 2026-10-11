"""Publication figures (static, print/light surface). Saved as 300-dpi PNG and vector PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order: slot 1, 2, 3
NEG = "#eb6834"  # diverging poles for signed coefficients
POS = "#2a78d6"
SEQ_CMAP = sns.blend_palette(["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"], as_cmap=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{name}.png"


def class_distribution(counts: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    names, values = list(counts), list(counts.values())
    bars = ax.bar(names, values, color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=2, color=INK_2, fontsize=8)
    ax.set_ylabel("Posts")
    ax.grid(axis="x", visible=False)
    ax.set_title("Class distribution after cleaning")
    return _save(fig, out_dir, "fig0_class_distribution")


def grouped_metric(metrics: pd.DataFrame, metric: str, ylabel: str, name: str, out_dir: Path,
                   fmt: str = "%.3f", unit_scale: bool = True) -> Path:
    """Grouped bars: one group per model, one bar per feature experiment."""
    models = [m for m in config.MODELS if m in set(metrics["model"])]
    exps = [e for e in config.EXPERIMENTS if e in set(metrics["experiment"])]
    x = np.arange(len(models))
    width = 0.8 / len(exps)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for j, exp in enumerate(exps):
        sub = metrics[metrics["experiment"] == exp].set_index("model").reindex(models)
        bars = ax.bar(x + (j - (len(exps) - 1) / 2) * width, sub[metric], width,
                      label=config.EXPERIMENTS[exp], color=SERIES[j], edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, fmt=fmt, padding=2, color=INK_2, fontsize=7)
    ax.set_xticks(x, [config.MODELS[m] for m in models])
    ax.set_ylabel(ylabel)
    top = metrics[metric].max() * 1.15
    ax.set_ylim(0, min(1.0, top) if unit_scale else top)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(exps))
    return _save(fig, out_dir, name)


def confusion_grid(matrices: dict, out_dir: Path) -> Path:
    """matrices: {(model, experiment): normalised ndarray}. Rows = experiments, columns = models."""
    models = [m for m in config.MODELS if any(k[0] == m for k in matrices)]
    exps = [e for e in config.EXPERIMENTS if any(k[1] == e for k in matrices)]
    short = ["Min", "Mild", "Mod", "Sev"]
    fig, axes = plt.subplots(len(exps), len(models), figsize=(2.4 * len(models), 2.4 * len(exps)),
                             squeeze=False)
    for i, e in enumerate(exps):
        for j, m in enumerate(models):
            ax = axes[i][j]
            cm = matrices.get((m, e))
            if cm is None:
                ax.axis("off")
                continue
            sns.heatmap(cm, ax=ax, vmin=0, vmax=1, cmap=SEQ_CMAP, annot=True, fmt=".2f",
                        cbar=False, square=True, linewidths=1, linecolor=SURFACE,
                        xticklabels=short, yticklabels=short, annot_kws={"fontsize": 7})
            ax.grid(False)  # rc grid would draw through the cells
            ax.set_title(f"{config.MODELS[m]} | {config.EXPERIMENTS[e].split(': ')[1]}", fontsize=8)
            ax.set_xlabel("Predicted" if i == len(exps) - 1 else "", fontsize=7)
            ax.set_ylabel("True" if j == 0 else "", fontsize=7)
            ax.tick_params(labelsize=7, length=0)
            ax.tick_params(axis="y", rotation=0)
    return _save(fig, out_dir, "fig2_confusion_matrices")


def signed_coefficients(df: pd.DataFrame, title: str, name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.12 * len(df))))
    colors = [POS if c > 0 else NEG for c in df["coefficient"]]
    ax.barh(df["feature"], df["coefficient"], color=colors, height=0.75)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Coefficient (Severe class)  ← lowers | raises →")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=6, colors=INK_2)
    ax.set_title(title)
    return _save(fig, out_dir, name)


def importances(df: pd.DataFrame, title: str, name: str, out_dir: Path) -> Path:
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.2 * len(df))))
    ax.barh(df["feature"], df["importance"], color=SERIES[0], height=0.7)
    ax.set_xlabel("Mean decrease in Gini impurity")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7, colors=INK_2)
    ax.set_title(title)
    return _save(fig, out_dir, name)


def timing(metrics: pd.DataFrame, out_dir: Path) -> Path:
    return grouped_metric(metrics, "fit_seconds", "Final-model training time (s)",
                          "fig5_training_time", out_dir, fmt="%.1f", unit_scale=False)
