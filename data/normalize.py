import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def normalize_column(dataframe: pd.DataFrame, column_index: int) -> pd.Series:
	"""Return z-score normalized values for a DataFrame column by index."""
	if not isinstance(column_index, int):
		raise TypeError("column_index must be an integer.")

	if column_index < -dataframe.shape[1] or column_index >= dataframe.shape[1]:
		raise IndexError(f"Column index {column_index} is out of range.")

	column_name = dataframe.columns[column_index]
	series = extract_numeric_column(dataframe, column_index)
	mean = series.mean(skipna=True)
	std = series.std(skipna=True, ddof=0) # ddof = 0 is the same as the std func for numpy

	if pd.isna(mean) or pd.isna(std):
		return pd.Series(np.nan, index=series.index, name=column_name)

	if std == 0:
		return pd.Series(0.0, index=series.index, name=column_name)

	return (series - mean) / std


def plot_value_counts(dataframe: pd.DataFrame, column_index: int) -> None:
	"""Plot value counts for a DataFrame column by index."""
	column_name = dataframe.columns[column_index]
	series = extract_numeric_column(dataframe, column_index).dropna()
	counts = series.value_counts().sort_index()

	ax = counts.plot(kind="bar", figsize=(10, 5))
	ax.set_title(f"Subset for The Persistence of Memory")
	ax.set_xlabel(column_name)
	ax.set_ylabel("Count")
	plt.tight_layout()
	plt.show()

def one_hot(
	dataframe: pd.DataFrame,
	column_index: int,
	categories: list[str],
	prefix: str | None = None,
) -> pd.DataFrame:
	"""Return one-hot columns for a comma-separated categorical column by index."""
	if not isinstance(column_index, int):
		raise TypeError("column_index must be an integer.")

	if column_index < -dataframe.shape[1] or column_index >= dataframe.shape[1]:
		raise IndexError(f"Column index {column_index} is out of range.")

	if not isinstance(categories, list) or any(not isinstance(cat, str) for cat in categories):
		raise TypeError("categories must be a list of strings.")

	if len(categories) == 0:
		raise ValueError("categories must not be empty.")

	if len(set(categories)) != len(categories):
		raise ValueError("categories must contain unique strings.")

	series = dataframe.iloc[:, column_index].fillna("").astype(str)
	split_values = series.str.split(",").apply(lambda items: [item for item in items if item])

	columns = [f"{prefix}_{cat}" if prefix else cat for cat in categories]
	dummies = pd.DataFrame(0, index=dataframe.index, columns=columns, dtype="int64")

	for category, output_col in zip(categories, columns):
		dummies[output_col] = split_values.apply(lambda items: int(category in items))

	return dummies

# import csv into pandas dataframe
df = pd.read_csv('data/training_data_202601_trainval.csv')
df_p = pd.read_csv('data/training_data_202601_trainval_pers.csv')
df_s = pd.read_csv('data/training_data_202601_trainval_star.csv')

df_train = pd.read_csv('data/training_data_202601_train.csv')
df_val = pd.read_csv('data/training_data_202601_val.csv')
df_test = pd.read_csv('data/training_data_202601_test.csv')


# Normalize columns 2, 4, 5, 6, 7, 8, 9, and 10 for train, val, and test datasets
for col_index in [2, 4, 5, 6, 7, 8, 9, 10]:
	column_name = df_train.columns[col_index]
	df_train[column_name] = normalize_column(df_train, col_index)

	column_name = df_val.columns[col_index]
	df_val[column_name] = normalize_column(df_val, col_index)

	column_name = df_test.columns[col_index]
	df_test[column_name] = normalize_column(df_test, col_index)

# Save the normalized datasets to new CSV files
df_train.to_csv("train_norm.csv", index=False)
df_val.to_csv("val_norm.csv", index=False)
df_test.to_csv("test_norm.csv", index=False)

# Create new columns for one-hot encoding of columns 11, 12, and 13 in the training dataset
room_categories = sorted(
	{
		item
		for value in df_train.iloc[:, 11].dropna().astype(str)
		for item in value.split(",")
		if item
	}
)

view_categories = sorted(
	{
		item
		for value in df_train.iloc[:, 12].dropna().astype(str)
		for item in value.split(",")
		if item
	}
)

season_categories = sorted(
	{
		item
		for value in df_train.iloc[:, 13].dropna().astype(str)
		for item in value.split(",")
		if item
	}
)

# Apply one-hot encoding to columns 11, 12, and 13 for train, val, and test datasets
for data in [df_train, df_val, df_test]:
	room_ohe = one_hot(data, 11, room_categories, prefix="room")
	view_ohe = one_hot(data, 12, view_categories, prefix="view")
	season_ohe = one_hot(data, 13, season_categories, prefix="season")

	# Keep original columns and append one-hot columns.
	if data is df_train:
		df_train = pd.concat([data, room_ohe, view_ohe, season_ohe], axis=1)
	elif data is df_val:
		df_val = pd.concat([data, room_ohe, view_ohe, season_ohe], axis=1)
	elif data is df_test:
		df_test = pd.concat([data, room_ohe, view_ohe, season_ohe], axis=1)

	# Optional: drop original multi-label string columns after encoding.
	# data = data.drop(data.columns[[11, 12, 13]], axis=1)

df_train.to_csv("train_norm.csv", index=False)
df_val.to_csv("val_norm.csv", index=False)
df_test.to_csv("test_norm.csv", index=False)







# print(df.iloc[:, [4, 5, 6, 7]]) 

# extract numeric values from column 4 and add as a new column
# df["col4_numeric"] = extract_numeric_column(df, 4)
# print(df[[df.columns[4], "col4_numeric"]].head())
# df["col4_zscore"] = normalize_column(df, 4)
# print(df[[df.columns[4], "col4_zscore"]].head())

# plot_value_counts(df, 5)
# plot_value_counts(df_p, 4)
# plot_value_counts(df_s, 4)
