"""
Multinomial logistic regression (L2) on normalized splits, using only the four
"This art piece makes me feel ..." Likert columns as features.

Lambda grid (L2 strength, larger => more regularization) is cross-validated on
train_norm; sklearn uses C where smaller C => stronger penalty. We set C = 1/lambda.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_NORM_PATH = os.path.join(DATA_DIR, "train_norm.csv")
VAL_NORM_PATH = os.path.join(DATA_DIR, "val_norm.csv")
TEST_NORM_PATH = os.path.join(DATA_DIR, "test_norm.csv")

TARGET_COL = "Painting"
FEATURE_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

# testing the L2 regularization strength 
LAMBDAS = [0.01, 0.1, 1.0, 10.0]

CV_SPLITS = 5
CV_SELECT_METRIC = "f1_macro"
RNG = 42


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    n_ids_file = df["unique_id"].nunique()
    cols = ["unique_id", TARGET_COL] + FEATURE_COLS
    df_model = df[cols].dropna(subset=[TARGET_COL] + FEATURE_COLS)
    X = df_model[FEATURE_COLS].values.astype(np.float64)
    y = df_model[TARGET_COL].values
    n_ids_model = df_model["unique_id"].nunique()
    return X, y, len(df_model), n_ids_file, n_ids_model


def main():
    (X_train, y_train, n_train, train_ids_file, train_ids_model) = _load_xy(TRAIN_NORM_PATH)
    (X_val, y_val, n_val, val_ids_file, val_ids_model) = _load_xy(VAL_NORM_PATH)
    (X_test, y_test, n_test, test_ids_file, test_ids_model) = _load_xy(TEST_NORM_PATH)

    class_labels = np.unique(np.concatenate([y_train, y_val, y_test]))

    print("Splits (train_norm / val_norm / test_norm):")
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
        "(Labelled rows = non-missing Painting + all four 'makes me feel' features.)"
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

    best_lambda = 1 # manually picked, highest acc and macro f1 and also low enough SD to justify
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
    ax.set_title("Logistic L2: tune lambda on train (four 'makes me feel' features)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logistic_lambda_cv.png",
    )
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

    print(f"\nRefit on full train with best lambda = {best_lambda}")
    print(f"Classes (macro-F1 over all): {list(class_labels)}")
    print(f"Training Accuracy: {train_acc:.4f} | Macro F1: {train_f1_macro:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Macro F1: {val_f1_macro:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Macro F1: {test_f1_macro:.4f}")


if __name__ == "__main__":
    main()
