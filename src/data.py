"""Dataset download, column detection, label normalisation and de-duplication."""
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

log = logging.getLogger(__name__)


def ensure_dataset(path: Path = config.RAW_CSV, url: str = config.DATA_URL) -> Path:
    """Download the public CSV once; reuse the local copy afterwards."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", url)
    urllib.request.urlretrieve(url, path)
    return path


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Map raw label values (names or ints, any case) to ordinal ids 0-3."""
    keys = labels.astype(str).str.strip().str.lower()
    mapped = keys.map(config.LABEL_ALIASES)
    unknown = sorted(keys[mapped.isna()].unique())
    if unknown:
        raise ValueError(f"Unrecognised label values: {unknown[:10]}")
    return mapped.astype(int)


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (text_col, label_col): label column is the one whose values are all known labels."""
    label_col = None
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip().str.lower()
        if len(values) and values.isin(config.LABEL_ALIASES.keys()).all():
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found among {list(df.columns)}")
    others = [c for c in df.columns if c != label_col]
    if not others:
        raise ValueError("No text column found")
    # Text column = remaining column with the longest average string.
    text_col = max(others, key=lambda c: df[c].astype(str).str.len().mean())
    return text_col, label_col


def prepare(df: pd.DataFrame, dedupe: bool = True) -> tuple[pd.DataFrame, dict]:
    """Standardise to [text, label], drop empty rows and (unless dedupe=False) conflicting/duplicate
    texts. Returns data and audit stats. dedupe=False reproduces the corpus as distributed."""
    text_col, label_col = detect_columns(df)
    out = pd.DataFrame({"text": df[text_col], "label": df[label_col]})
    stats = {"raw_rows": len(out), "text_col": text_col, "label_col": label_col}

    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""]
    stats["dropped_empty"] = stats["raw_rows"] - len(out)

    out["label"] = normalize_labels(out["label"])
    stats["dropped_conflicting"] = stats["dropped_duplicates"] = 0
    if not dedupe:
        return _finish(out, stats)

    # Same text with different labels is ambiguous ground truth: drop every copy.
    labels_per_text = out.groupby("text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    before = len(out)
    out = out[~out["text"].isin(conflicting)]
    stats["dropped_conflicting"] = before - len(out)

    before = len(out)
    out = out.drop_duplicates(subset="text", keep="first")
    stats["dropped_duplicates"] = before - len(out)
    return _finish(out, stats)


def _finish(out: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, dict]:
    out = out.reset_index(drop=True)
    stats["final_rows"] = len(out)
    stats["class_counts"] = {
        config.CLASS_NAMES[k]: int(v) for k, v in out["label"].value_counts().sort_index().items()
    }
    return out, stats
