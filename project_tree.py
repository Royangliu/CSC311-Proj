import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import matplotlib.pyplot as plt
import json

# Keep feature selection identical to train_tree.py.
FEATURE_COLS = [
	2, 4, 5, 6, 7, 8, 9, 10,
	16, 17, 18, 19, 20, 21,
	22, 23, 24, 25,
	26, 27, 28, 29,
]
TARGET_COL = 1


def build_features(df: pd.DataFrame) -> np.ndarray:
	return np.stack([df.iloc[:, col] for col in FEATURE_COLS], axis=1)


def main() -> None:
	train_df = pd.read_csv("data/train_norm.csv")
	val_df = pd.read_csv("data/val_norm.csv")
	test_df = pd.read_csv("data/test_norm.csv")

	train_val_df = pd.concat([train_df, val_df], ignore_index=True)

	x_train = build_features(train_val_df)
	y_train = train_val_df.iloc[:, TARGET_COL]

	x_val = build_features(val_df)
	y_val = val_df.iloc[:, TARGET_COL]

	x_test = build_features(test_df)
	y_test = test_df.iloc[:, TARGET_COL]

	base_model = RandomForestClassifier(
		criterion="entropy",
		random_state=42,
		n_jobs=-1,
	)

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	param_grid = {
		"n_estimators": [300, 400, 500],
		"max_depth": [7, 8, 9, 10],
		"min_samples_split": [2],
		"min_samples_leaf": [2],
		"max_features": ["sqrt"],
		"class_weight": ["balanced_subsample"],
	}

	search = GridSearchCV(
		estimator=base_model,
		param_grid=param_grid,
		scoring="accuracy",
		cv=cv,
		n_jobs=-1,
		verbose=1,
	)
	search.fit(x_train, y_train)
	model = search.best_estimator_
	best_idx = search.best_index_
	best_cv_mean = search.cv_results_["mean_test_score"][best_idx]
	best_cv_std = search.cv_results_["std_test_score"][best_idx]
	best_fold_scores = [
		search.cv_results_[f"split{i}_test_score"][best_idx]
		for i in range(cv.get_n_splits())
	]

	model.fit(x_train, y_train)

	train_pred = model.predict(x_train)
	val_pred = model.predict(x_val)
	test_pred = model.predict(x_test)

	print("RandomForest trained with feature columns:", FEATURE_COLS)
	print("Best Params:", search.best_params_)
	print(f"Best 5-Fold CV Accuracy (mean): {best_cv_mean:.4f}")
	print(f"Best 5-Fold CV Accuracy (std):  {best_cv_std:.4f}")
	print("Best 5-Fold CV fold scores:", [f"{score:.4f}" for score in best_fold_scores])
	print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
	print(f"Val Accuracy:   {accuracy_score(y_val, val_pred):.4f}")
	print(f"Test Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
	print(f"Train Macro-F1: {f1_score(y_train, train_pred, average='macro'):.4f}")
	print(f"Val Macro-F1:   {f1_score(y_val, val_pred, average='macro'):.4f}")
	print(f"Test Macro-F1:  {f1_score(y_test, test_pred, average='macro'):.4f}")

	# Generate and save confusion matrix for validation set
	val_conf_matrix = confusion_matrix(y_val, val_pred)
	plots_dir = Path("plots")
	plots_dir.mkdir(parents=True, exist_ok=True)

	# Generate and save confusion matrix for test set
	test_conf_matrix = confusion_matrix(y_test, test_pred)
	plots_dir = Path("plots")
	plots_dir.mkdir(parents=True, exist_ok=True)

	plt.figure()
	disp = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=["Persistence", "Starry Night", "Water Lilies"])
	disp.plot(cmap="Blues", values_format="d")
	plt.title("Test Confusion Matrix (Random Forest Seed=42)")
	plt.tight_layout()
	plt.savefig(plots_dir / "test_confusion_matrix.png", dpi=300)
	plt.show()


	forest = []

	for estimator in model.estimators_:
		tree = estimator.tree_

		tree_dict = {
			"feature": tree.feature.tolist(),
			"threshold": tree.threshold.tolist(),
			"left": tree.children_left.tolist(),
			"right": tree.children_right.tolist(),
			"value": tree.value.tolist()
		}

		forest.append(tree_dict)

	with open("forest.json", "w") as f:
		json.dump(forest, f)


if __name__ == "__main__":
	main()

