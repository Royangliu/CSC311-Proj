"""
Multinomial logistic regression (L2) on train_norm / val_norm / test_norm.

Features: every numeric column in those CSVs except ``unique_id``,
``Painting``, and willingness-to-pay (price). That leaves intensity, Likert,
colours, objects, room/view/season indicators. Free-text fields stay strings
in ``*_norm.csv`` and are skipped—logistic regression needs numeric inputs.

Lambda grid is cross-validated on train; sklearn uses C with C = 1/lambda.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
TRAIN_NORM_PATH = os.path.join(DATA_DIR, "train_norm.csv")
VAL_NORM_PATH = os.path.join(DATA_DIR, "val_norm.csv")
TEST_NORM_PATH = os.path.join(DATA_DIR, "test_norm.csv")

TARGET_COL = "Painting"
PRICE_COL = (
    "How much (in Canadian dollars) would you be willing to pay for this painting?"
)
EXCLUDE_COLS = {"unique_id", TARGET_COL, PRICE_COL}

LAMBDAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
CV_SPLITS = 5
CV_SELECT_METRIC = "f1_macro"  # or accuracy
RNG = 42

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


def _numeric_feature_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _load_xy(csv_path: str, feature_cols: list):
    df = pd.read_csv(csv_path)
    n_ids_file = df["unique_id"].nunique()
    cols = ["unique_id", TARGET_COL] + feature_cols
    df_model = df[cols].dropna(subset=[TARGET_COL] + feature_cols)
    X = df_model[feature_cols].values.astype(np.float64)
    y = df_model[TARGET_COL].values
    n_ids_model = df_model["unique_id"].nunique()
    return X, y, len(df_model), n_ids_file, n_ids_model


def main():
    schema_df = pd.read_csv(TRAIN_NORM_PATH)
    feature_cols = _numeric_feature_columns(schema_df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found (check train_norm.csv).")

    (X_train, y_train, n_train, train_ids_file, train_ids_model) = _load_xy(
        TRAIN_NORM_PATH, feature_cols
    )
    (X_val, y_val, n_val, val_ids_file, val_ids_model) = _load_xy(
        VAL_NORM_PATH, feature_cols
    )
    (X_test, y_test, n_test, test_ids_file, test_ids_model) = _load_xy(
        TEST_NORM_PATH, feature_cols
    )

    class_labels = np.unique(np.concatenate([y_train, y_val, y_test]))

    print(
        f"Features: {len(feature_cols)} numeric columns from train_norm "
        "(all numeric except unique_id, Painting, price, and non-numeric text)."
    )
    for i, c in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {c}")

    print("\nSplits (train_norm / val_norm / test_norm):")
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
    print(
        "(Labelled rows = non-missing Painting + every numeric feature; "
        "empty NAs dropped.)"
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RNG)
    cv_acc_mean, cv_acc_std = [], []
    cv_f1_mean, cv_f1_std = [], []

    for lam in LAMBDAS:
        C = 1.0 / lam
        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=5000,
        )
        out = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
        )
        cv_acc_mean.append(out["test_accuracy"].mean())
        cv_acc_std.append(out["test_accuracy"].std())
        cv_f1_mean.append(out["test_f1_macro"].mean())
        cv_f1_std.append(out["test_f1_macro"].std())

    if CV_SELECT_METRIC == "accuracy":
        best_idx = int(np.argmax(cv_acc_mean))
    elif CV_SELECT_METRIC == "f1_macro":
        best_idx = int(np.argmax(cv_f1_mean))
    else:
        raise ValueError("CV_SELECT_METRIC must be 'accuracy' or 'f1_macro'")

    best_lambda = 10
    best_C = 1.0 / best_lambda
    best_acc_idx = int(np.argmax(cv_acc_mean))
    best_f1_idx = int(np.argmax(cv_f1_mean))

    print(
        "\nCross-val on train (StratifiedKFold): mean accuracy & macro F1 "
        "(same folds for both)"
    )
    print("lambda | C      | mean acc ± std    | mean f1_macro ± std")
    for lam, C, am, ast, fm, fst in zip(
        LAMBDAS,
        [1.0 / l for l in LAMBDAS],
        cv_acc_mean,
        cv_acc_std,
        cv_f1_mean,
        cv_f1_std,
    ):
        print(
            f"{lam:>6g} | {C:>6.4g} | {am:.4f} ± {ast:.4f} | {fm:.4f} ± {fst:.4f}"
        )

    print(f"\nBest lambda by CV accuracy: {LAMBDAS[best_acc_idx]} (mean acc {cv_acc_mean[best_acc_idx]:.4f})")
    print(f"Best lambda by CV macro F1:  {LAMBDAS[best_f1_idx]} (mean F1  {cv_f1_mean[best_f1_idx]:.4f})")
    print(
        f"\nRefit uses lambda = {best_lambda} (highest CV {CV_SELECT_METRIC}; C = {best_C:g})"
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LAMBDAS))
    ax.errorbar(
        x,
        cv_acc_mean,
        yerr=cv_acc_std,
        fmt="o-",
        capsize=4,
        label="CV accuracy",
    )
    ax.errorbar(
        x,
        cv_f1_mean,
        yerr=cv_f1_std,
        fmt="s--",
        capsize=4,
        label="CV macro F1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in LAMBDAS])
    ax.set_xlabel("lambda (L2)")
    ax.set_ylabel("Mean CV score")
    ax.set_title(f"Logistic L2: all {len(feature_cols)} numeric train_norm features")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(ROOT_DIR, "logistic_lambda_cv.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    final_clf = LogisticRegression(
        penalty="l2",
        C=best_C,
        solver="lbfgs",
        max_iter=5000,
    )
    final_clf.fit(X_train, y_train)

    train_acc = final_clf.score(X_train, y_train)
    val_acc = final_clf.score(X_val, y_val)
    test_acc = final_clf.score(X_test, y_test)

    f1_kw = dict(labels=class_labels, average="macro", zero_division=0)
    train_f1_macro = f1_score(y_train, final_clf.predict(X_train), **f1_kw)
    val_f1_macro = f1_score(y_val, final_clf.predict(X_val), **f1_kw)
    test_f1_macro = f1_score(y_test, final_clf.predict(X_test), **f1_kw)

    print(f"\nRefit on full train with lambda = {best_lambda} (CV-selected)")
    print(f"Classes (macro-F1 over all): {list(class_labels)}")
    print(f"Training Accuracy: {train_acc:.4f} | Macro F1: {train_f1_macro:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Macro F1: {val_f1_macro:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Macro F1: {test_f1_macro:.4f}")

    y_val_hat = final_clf.predict(X_val)
    y_test_hat = final_clf.predict(X_test)
    _save_confusion_matrix_png(
        y_val,
        y_val_hat,
        class_labels,
        f"Logistic L2 (λ={best_lambda}) — validation",
        os.path.join(PLOTS_DIR, "confusion_logistic_validation.png"),
    )
    _save_confusion_matrix_png(
        y_test,
        y_test_hat,
        class_labels,
        f"Logistic L2 (λ={best_lambda}) — test",
        os.path.join(PLOTS_DIR, "confusion_logistic_test.png"),
    )


if __name__ == "__main__":
    main()
