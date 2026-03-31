import json
import re
import pandas as pd

# Load forest from JSON
with open("forest.json", "r", encoding="utf-8") as f:
	FOREST_DATA = json.load(f)

# Backward compatibility:
# - New format: {"feature_cols": [...], "classes": [...], "trees": [...]}
# - Old format: [{"feature": ..., "left": ..., "right": ..., "value": ...}, ...]
if isinstance(FOREST_DATA, dict):
	required_keys = {"feature_cols", "classes", "trees"}
	missing = required_keys - set(FOREST_DATA.keys())
	if missing:
		raise ValueError(f"forest.json is missing required keys: {sorted(missing)}")

	FEATURE_COLS = FOREST_DATA["feature_cols"]
	CLASSES = FOREST_DATA["classes"]
	TREES = FOREST_DATA["trees"]

	if not isinstance(FEATURE_COLS, list) or not FEATURE_COLS:
		raise ValueError("forest.json key 'feature_cols' must be a non-empty list")
	if not isinstance(CLASSES, list) or not CLASSES:
		raise ValueError("forest.json key 'classes' must be a non-empty list")
	if not isinstance(TREES, list) or not TREES:
		raise ValueError("forest.json key 'trees' must be a non-empty list")
else:
	FEATURE_COLS = [
		2, 4, 5, 6, 7, 8, 9, 10,
		16, 17, 18, 19, 20, 21,
		22, 23, 24, 25,
		26, 27, 28, 29,
	]
	CLASSES = None
	TREES = FOREST_DATA

# # Train-split mean / population std (ddof=0) for columns 2,4,5,6,7,8,9,10 — same as data/normalize.py
# _FEATURE_MEANS = (
# 	6.334375,
# 	2.7198211624441133,
# 	3.4172876304023845,
# 	3.700447093889717,
# 	2.4053651266766023,
# 	3.825633383010432,
# 	3.782089552238806,
# 	4300023.769325337,
# )
# _FEATURE_STDS = (
# 	2.11670507704416,
# 	1.3894033600003164,
# 	1.2983765304160497,
# 	1.2418376519929453,
# 	1.4339614475296873,
# 	2.5228018588062273,
# 	2.6760923490487234,
# 	54001013.48960381,
# )
_FEATURE_MEANS = None
_FEATURE_STDS = None

EXPECTED_LABELS = [
	"The Persistence of Memory",
	"The Starry Night",
	"The Water Lily Pond",
]

LABEL_ALIASES = {
	"persistence": "The Persistence of Memory",
	"the persistence of memory": "The Persistence of Memory",
	"starry night": "The Starry Night",
	"the starry night": "The Starry Night",
	"water lilies": "The Water Lily Pond",
	"water lily pond": "The Water Lily Pond",
	"the water lily pond": "The Water Lily Pond",
}


def _looks_like_target_label(value) -> bool:
	"""Return True if a value appears to be one of the three target labels."""
	if pd.isna(value):
		return False
	key = str(value).strip().lower()
	return key in LABEL_ALIASES


def _infer_index_shift(df: pd.DataFrame) -> int:
	"""Infer positional index shift: -1 when target column is omitted, otherwise 0."""
	if df.shape[1] <= 1:
		return -1

	# If column 1 looks like class labels, keep original training indices.
	sample = df.iloc[:, 1].dropna().head(25)
	if not sample.empty and any(_looks_like_target_label(v) for v in sample):
		return 0
	return -1

def one_hot(
	dataframe: pd.DataFrame,
	column_index: int,
	categories: list[str],
	prefix: str | None = None,
) -> pd.DataFrame:
	"""Return one-hot columns for a comma-separated categorical column by index."""
	series = dataframe.iloc[:, column_index].fillna("").astype(str)
	split_values = series.str.split(",").apply(lambda items: [item for item in items if item])

	columns = [f"{prefix}_{cat}" if prefix else cat for cat in categories]
	dummies = pd.DataFrame(0, index=dataframe.index, columns=columns, dtype="int64")

	for category, output_col in zip(categories, columns):
		dummies[output_col] = split_values.apply(lambda items: int(category in items))

	return dummies


def extract_numeric_column(dataframe: pd.DataFrame, column_index: int) -> pd.Series:
	"""Extract the first numeric value from each cell in a DataFrame column by index."""
	if not isinstance(column_index, int):
		raise TypeError("column_index must be an integer.")

	if column_index < -dataframe.shape[1] or column_index >= dataframe.shape[1]:
		raise IndexError(f"Column index {column_index} is out of range.")

	column_name = dataframe.columns[column_index]
	series = dataframe.iloc[:, column_index].astype(str)
	extracted = series.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
	return pd.to_numeric(extracted, errors="coerce").rename(column_name)


def replace_column_with_numeric(
	dataframe: pd.DataFrame,
	column_index: int,
	inplace: bool = False,
) -> pd.DataFrame:
	"""Replace a column with extract_numeric_column output and return the updated DataFrame."""
	if not isinstance(column_index, int):
		raise TypeError("column_index must be an integer.")

	if column_index < -dataframe.shape[1] or column_index >= dataframe.shape[1]:
		raise IndexError(f"Column index {column_index} is out of range.")

	column_name = dataframe.columns[column_index]
	target_df = dataframe if inplace else dataframe.copy()
	target_df[column_name] = extract_numeric_column(dataframe, column_index)
	return target_df


def _traverse_tree(tree, features):
	"""
	Traverse a single decision tree and return leaf vote counts.
	"""
	node = 0
	while tree["left"][node] != -1:
		feat_idx = tree["feature"][node]
		threshold = tree["threshold"][node]
		if features[feat_idx] <= threshold:
			node = tree["left"][node]
		else:
			node = tree["right"][node]
	# Return vote counts at this leaf
	leaf_votes = tree["value"][node]
	while isinstance(leaf_votes, list) and len(leaf_votes) == 1 and isinstance(leaf_votes[0], list):
		leaf_votes = leaf_votes[0]
	return leaf_votes


# def normalize_with_params(series: pd.Series, mean: float, std: float, column_name: str) -> pd.Series:
# 	"""Apply z-score normalization using externally provided mean and std."""
# 	if pd.isna(mean) or pd.isna(std):
# 		return pd.Series(np.nan, index=series.index, name=column_name)

# 	if std == 0:
# 		return pd.Series(0.0, index=series.index, name=column_name)

# 	return ((series - mean) / std).rename(column_name)



def predict_one(row_features):
	"""
	Predict class for a single feature vector (already extracted).
	"""
	num_classes = len(CLASSES) if CLASSES is not None else len(_traverse_tree(TREES[0], row_features))
	votes = [0] * num_classes
	for tree in TREES:
		leaf_votes = _traverse_tree(tree, row_features)
		for i, count in enumerate(leaf_votes):
			votes[i] += count
	best_idx = votes.index(max(votes))
	if CLASSES is None:
		return best_idx
	return CLASSES[best_idx]


def _coerce_feature_value(value):
	"""Convert a feature value to float, defaulting to 0.0 for missing/unparseable values."""
	if pd.isna(value):
		return 0.0
	if isinstance(value, (int, float)):
		return float(value)
	text = str(value).strip()
	if not text:
		return 0.0
	try:
		return float(text)
	except ValueError:
		match = re.search(r"[-+]?\d*\.?\d+", text)
		return float(match.group(0)) if match else 0.0


def _normalize_label(label):
	"""Map legacy/variant labels to the checker's canonical labels."""
	key = str(label).strip().lower()
	return LABEL_ALIASES.get(key, label)


def predict(row, feature_cols=None):
	"""
	Helper function to make prediction for a given input row.
	Extracts features from the raw row and calls predict_one.
	"""
	if feature_cols is None:
		feature_cols = FEATURE_COLS

	# Robust extraction: if model expects columns not present in the input row, use 0.0.
	features = []
	num_cols = len(row)
	for col in feature_cols:
		if isinstance(col, int) and -num_cols <= col < num_cols:
			features.append(_coerce_feature_value(row.iloc[col]))
		else:
			features.append(0.0)
	if _FEATURE_MEANS is not None and _FEATURE_STDS is not None:
		# Z-score the eight numeric survey columns (indices 2,4-10) before tree traversal.
		for i in range(8):
			x = features[i]
			if pd.isna(x):
				features[i] = 0.0
				continue
			m, s = _FEATURE_MEANS[i], _FEATURE_STDS[i]
			features[i] = 0.0 if s == 0 else (float(x) - m) / s
	return _normalize_label(predict_one(features))

# Note that the python translation of the forest may result in floating point errors that causes slight changes to class assignments for some edge cases. 
def predict_all(filename):
	"""
	Make predictions for the data in filename.
	"""
	global CLASSES

	# Read the file containing the test data
	df = pd.read_csv(filename)
	index_shift = _infer_index_shift(df)
	effective_feature_cols = [
		col + index_shift if isinstance(col, int) and col >= 2 else col
		for col in FEATURE_COLS
	]
	if CLASSES is None:
		# Legacy forest format fallback when class names are not stored in forest.json.
		if index_shift == 0:
			CLASSES = sorted(df.iloc[:, 1].dropna().unique().tolist())
		else:
			CLASSES = EXPECTED_LABELS.copy()

	room_categories = sorted(["Bathroom", "Bedroom", "Dining room", "Living room", "Office"])

	view_categories = sorted(["By yourself", "Coworkers/Classmates", "Family members", "Friends", "Strangers"])

	season_categories = sorted(["Fall", "Spring", "Summer", "Winter"])

	# Apply multi-hot encoding to columns 11, 12, and 13 for train, val, and test datasets
	room_ohe = one_hot(df, 11 + index_shift, room_categories, prefix="room")
	view_ohe = one_hot(df, 12 + index_shift, view_categories, prefix="view")
	season_ohe = one_hot(df, 13 + index_shift, season_categories, prefix="season")

	df = pd.concat([df, room_ohe, view_ohe, season_ohe], axis=1)

	for col_index in [4 + index_shift, 5 + index_shift, 6 + index_shift, 7 + index_shift, 10 + index_shift]:
		if -df.shape[1] <= col_index < df.shape[1]:
			df = replace_column_with_numeric(df, col_index, inplace=True)

	predictions = []
	for idx, row in df.iterrows():
		pred = predict(row, feature_cols=effective_feature_cols)
		predictions.append(pred)

	return predictions

if __name__ == "__main__":
	filename = "data/training_data_202601_test.csv"
	predictions = predict_all(filename)

	df = pd.read_csv(filename)
	true_labels = df.iloc[:, 1].tolist()

	correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
	percentage_correct = 100.0 * correct / len(true_labels) if true_labels else 0.0

	print(f"Correct: {correct}/{len(true_labels)}")
	print(f"Percentage correct: {percentage_correct:.2f}%")
	# print(predictions)