"""Print `_TRAIN = (...)` for pasting into `pred_knn.py` after ``_FEATURE_STDS``.

If ``normalize.py`` or the train split changes, refresh ``_FEATURE_MEANS`` /
``_FEATURE_STDS`` in ``pred_knn.py`` too (same as columns 2,4,5,6,7,8,9 on
``training_data_202601_train.csv``).
"""

from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "train_norm.csv")

TARGET_COL = "Painting"
INTENSITY_COL = (
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
)
LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]
COLOURS_COL = "How many prominent colours do you notice in this painting?"
OBJECTS_COL = "How many objects caught your eye in the painting?"
FEATURE_COLS = [INTENSITY_COL] + LIKERT_COLS + [COLOURS_COL, OBJECTS_COL]


def _parse_feat(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    rows = []
    with open(TRAIN_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get(TARGET_COL) or "").strip()
            if not lab:
                continue
            feats = []
            skip = False
            for key in FEATURE_COLS:
                v = _parse_feat(row.get(key))
                if v is None:
                    skip = True
                    break
                feats.append(v)
            if skip:
                continue
            rows.append((tuple(feats), lab))

    print("# Frozen training set: snapshot of project_knn_all labelled train_norm rows")
    print(f"# ({len(rows)} rows). Paste into pred_knn.py after _CLASS_ORDER.")
    print("_TRAIN = (")
    for t, lab in rows:
        f0, f1, f2, f3, f4, f5, f6 = t
        print(
            f"    (({f0!r}, {f1!r}, {f2!r}, {f3!r}, {f4!r}, {f5!r}, {f6!r}), {lab!r}),"
        )
    print(")")


if __name__ == "__main__":
    main()
