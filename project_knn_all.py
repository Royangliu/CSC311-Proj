"""
k-NN on normalized train/val/test CSVs with all scalar survey features used in
project_knn_likert plus the emotion-intensity scale (z-scored in *_norm.csv):

  - Intensity (1–10 question)
  - Four "This art piece makes me feel ..." Likert columns
  - Prominent colours
  - Objects count

Same workflow as project_knn_likert.py; tune k from the plot, then set SELECTED_K
and NEIGHBOR_WEIGHTS.

Optional bag-of-words columns in ``train_norm.csv`` (``bow_*`` prefixes) are **off**
by default. Set ``INCLUDE_BOW_FEATURES = True`` to add a subset: the top
``BOW_TOP_N`` tokens by training-set frequency (``BOW_TOP_N = None`` uses all BOW).

``weights='uniform'`` (default): each of the k neighbors counts equally.
``weights='distance'``: closer neighbors count more (inverse distance).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
TRAIN_NORM_PATH = os.path.join(DATA_DIR, "train_norm.csv")
VAL_NORM_PATH = os.path.join(DATA_DIR, "val_norm.csv")
TEST_NORM_PATH = os.path.join(DATA_DIR, "test_norm.csv")

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
BASE_FEATURE_COLS = [INTENSITY_COL] + LIKERT_COLS + [COLOURS_COL, OBJECTS_COL]

INCLUDE_BOW_FEATURES = False
BOW_TOP_N: int | None = 100

BOW_PREFIXES = ("bow_feel_", "bow_food_", "bow_music_")

SELECTED_K = 19
K_MAX = 30

NEIGHBOR_WEIGHTS = "uniform"
_WEIGHT_CHOICES = ("uniform", "distance")

_CM_LABEL_SHORT = {
    "The Persistence of Memory": "Persistence",
    "The Starry Night": "Starry Night",
    "The Water Lily Pond": "Water Lilies",
}


def _save_confusion_matrix_png(y_true, y_pred, class_labels, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels))
    short = [_CM_LABEL_SHORT.get(c, c) for c in class_labels]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")


def _bow_columns_sorted(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(BOW_PREFIXES))


def _feature_columns(train_df: pd.DataFrame) -> list[str]:
    for c in BASE_FEATURE_COLS:
        if c not in train_df.columns:
            raise KeyError(f"Missing expected column {c!r} in training CSV")
    cols = list(BASE_FEATURE_COLS)
    if not INCLUDE_BOW_FEATURES:
        return cols
    bow_all = _bow_columns_sorted(train_df)
    if not bow_all:
        return cols
    if BOW_TOP_N is None:
        cols.extend(bow_all)
        return cols
    n = min(BOW_TOP_N, len(bow_all))
    sums = train_df[bow_all].sum(numeric_only=True).sort_values(ascending=False)
    cols.extend(list(sums.head(n).index))
    return cols


def _load_xy_from_df(df: pd.DataFrame, feature_cols: list[str], label: str):
    for c in feature_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} in {label}")
    n_ids_file = df["unique_id"].nunique()
    need = ["unique_id", TARGET_COL] + feature_cols
    df_model = df[need].dropna(subset=[TARGET_COL] + feature_cols)
    X = df_model[feature_cols].values.astype(np.float64)
    y = df_model[TARGET_COL].values
    n_ids_model = df_model["unique_id"].nunique()
    return X, y, len(df_model), n_ids_file, n_ids_model


def _load_xy(csv_path: str, feature_cols: list[str]):
    return _load_xy_from_df(pd.read_csv(csv_path), feature_cols, csv_path)


def main():
    train_df = pd.read_csv(TRAIN_NORM_PATH)
    feature_cols = _feature_columns(train_df)

    (X_train, y_train, n_train, train_ids_file, train_ids_model) = _load_xy_from_df(
        train_df, feature_cols, TRAIN_NORM_PATH
    )
    (X_val, y_val, n_val, val_ids_file, val_ids_model) = _load_xy(
        VAL_NORM_PATH, feature_cols
    )
    (X_test, y_test, n_test, test_ids_file, test_ids_model) = _load_xy(
        TEST_NORM_PATH, feature_cols
    )

    class_labels = np.unique(np.concatenate([y_train, y_val, y_test]))

    bow_used = sum(1 for c in feature_cols if c.startswith(BOW_PREFIXES))
    print(
        "Splits (train_norm / val_norm / test_norm), "
        f"{len(feature_cols)} features "
        f"({len(BASE_FEATURE_COLS)} base + {bow_used} BOW):"
    )
    print(
        f"  INCLUDE_BOW_FEATURES={INCLUDE_BOW_FEATURES!r}, "
        f"BOW_TOP_N={BOW_TOP_N!r}"
    )
    print(
        f"  Train: {n_train} labelled rows | unique_id in file={train_ids_file}, "
        f"among labelled rows={train_ids_model}"
    )
    print(
        f"  Val:   {n_val} labelled rows | unique_id in file={val_ids_file}, "
        f"among labelled rows={val_ids_model}"
    )
    print(
        f"  Test:  {n_test} labelled rows | unique_id in file={test_ids_file}, "
        f"among labelled rows={test_ids_model}"
    )
    print("(Labelled rows need non-missing Painting + every feature column.)")

    max_k = min(K_MAX, len(X_train))
    k_values = list(range(1, max_k + 1))
    val_losses_by_weight = {w: [] for w in _WEIGHT_CHOICES}

    for w in _WEIGHT_CHOICES:
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, weights=w)
            knn.fit(X_train, y_train)
            val_acc = knn.score(X_val, y_val)
            val_losses_by_weight[w].append(1.0 - val_acc)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for w in _WEIGHT_CHOICES:
        ax.plot(
            k_values,
            val_losses_by_weight[w],
            marker="o" if w == "uniform" else "s",
            markersize=4,
            label=f"weights={w!r}",
        )
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Validation loss (1 − accuracy)")
    ax.set_title("k-NN: intensity + Likert + colours + objects — val loss vs k & weights")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values[:: max(1, len(k_values) // 15)])

    plot_path = os.path.join(ROOT_DIR, "k_vs_val_loss_knn_all.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    for w in _WEIGHT_CHOICES:
        losses = val_losses_by_weight[w]
        best_i = int(np.argmin(losses))
        print(
            f"weights={w!r}: best k = {k_values[best_i]} "
            f"(min val loss {losses[best_i]:.4f})"
        )

    print("\nk | val loss uniform | val loss distance")
    for k, lu, ld in zip(
        k_values,
        val_losses_by_weight["uniform"],
        val_losses_by_weight["distance"],
    ):
        print(f"{k:3d} | {lu:.4f}           | {ld:.4f}")

    if SELECTED_K < 1 or SELECTED_K > len(X_train):
        raise ValueError(
            f"SELECTED_K={SELECTED_K} is invalid; must be in [1, {len(X_train)}] "
            "(n_neighbors cannot exceed training set size)."
        )
    if NEIGHBOR_WEIGHTS not in _WEIGHT_CHOICES:
        raise ValueError(
            f"NEIGHBOR_WEIGHTS={NEIGHBOR_WEIGHTS!r} must be one of {_WEIGHT_CHOICES}"
        )

    print(f"\nAt k = {SELECTED_K} (compare weights):")
    for w in _WEIGHT_CHOICES:
        knn = KNeighborsClassifier(n_neighbors=SELECTED_K, weights=w)
        knn.fit(X_train, y_train)
        print(
            f"  weights={w!r}: val acc {knn.score(X_val, y_val):.4f} | "
            f"test acc {knn.score(X_test, y_test):.4f}"
        )

    final_knn = KNeighborsClassifier(n_neighbors=SELECTED_K, weights=NEIGHBOR_WEIGHTS)
    final_knn.fit(X_train, y_train)

    train_acc = final_knn.score(X_train, y_train)
    val_acc = final_knn.score(X_val, y_val)
    test_acc = final_knn.score(X_test, y_test)

    f1_kw = dict(labels=class_labels, average="macro", zero_division=0)
    train_f1_macro = f1_score(y_train, final_knn.predict(X_train), **f1_kw)
    val_f1_macro = f1_score(y_val, final_knn.predict(X_val), **f1_kw)
    test_f1_macro = f1_score(y_test, final_knn.predict(X_test), **f1_kw)

    print(f"\nUsing SELECTED_K = {SELECTED_K}, NEIGHBOR_WEIGHTS = {NEIGHBOR_WEIGHTS!r}")
    print(f"Classes (macro-F1 over all): {list(class_labels)}")
    print(f"Training Accuracy: {train_acc:.4f} | Macro F1: {train_f1_macro:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Macro F1: {val_f1_macro:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Macro F1: {test_f1_macro:.4f}")

    y_val_hat = final_knn.predict(X_val)
    y_test_hat = final_knn.predict(X_test)
    _save_confusion_matrix_png(
        y_val,
        y_val_hat,
        class_labels,
        f"k-NN (k={SELECTED_K}, {NEIGHBOR_WEIGHTS}) — validation",
        os.path.join(PLOTS_DIR, "confusion_knn_all_validation.png"),
    )
    _save_confusion_matrix_png(
        y_test,
        y_test_hat,
        class_labels,
        f"k-NN (k={SELECTED_K}, {NEIGHBOR_WEIGHTS}) — test",
        os.path.join(PLOTS_DIR, "confusion_knn_all_test.png"),
    )


if __name__ == "__main__":
    main()
