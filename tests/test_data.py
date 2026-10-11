import pandas as pd
import pytest

from src.data import detect_columns, normalize_labels, prepare


def test_normalize_labels_accepts_names_ints_and_case():
    s = pd.Series(["Minimum", "MILD", " moderate ", "severe", 0, "3"])
    assert list(normalize_labels(s)) == [0, 1, 2, 3, 0, 3]


def test_normalize_labels_rejects_unknown():
    with pytest.raises(ValueError, match="Unrecognised"):
        normalize_labels(pd.Series(["mild", "extreme"]))


def test_detect_columns_any_order():
    df = pd.DataFrame({"label": ["mild", "severe"], "post": ["long text here", "more text"]})
    assert detect_columns(df) == ("post", "label")
    assert detect_columns(df[["post", "label"]]) == ("post", "label")


def test_detect_columns_without_label_column():
    with pytest.raises(ValueError, match="No label column"):
        detect_columns(pd.DataFrame({"a": ["x"], "b": ["y"]}))


def test_prepare_drops_empty_conflicting_and_duplicates():
    raw = pd.DataFrame({
        "text": ["same", "same", "conflict", "conflict", "  ", None, "ok"],
        "label": ["mild", "mild", "mild", "severe", "mild", "mild", "severe"],
    })
    df, stats = prepare(raw)
    assert list(df["text"]) == ["same", "ok"]
    assert list(df["label"]) == [1, 3]
    assert stats["dropped_empty"] == 2
    assert stats["dropped_conflicting"] == 2
    assert stats["dropped_duplicates"] == 1
    assert stats["final_rows"] == 2
    assert stats["class_counts"] == {"Mild": 1, "Severe": 1}


def test_prepare_without_dedupe_keeps_duplicates_and_conflicts():
    raw = pd.DataFrame({"text": ["a", "a", "b", "b", ""], "label": ["mild", "mild", "mild", "severe", "mild"]})
    df, stats = prepare(raw, dedupe=False)
    assert len(df) == 4
    assert stats["dropped_empty"] == 1
    assert stats["dropped_conflicting"] == 0 and stats["dropped_duplicates"] == 0
