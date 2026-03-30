"""
Random Forest predictor loaded from forest.json.
No sklearn imports - pure Python tree traversal.
"""

import json
import pandas as pd

# Load forest from JSON
with open("forest.json", "r") as f:
    FOREST_DATA = json.load(f)

FEATURE_COLS = FOREST_DATA["feature_cols"]
CLASSES = FOREST_DATA["classes"]
TREES = FOREST_DATA["trees"]


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
    return tree["value"][node]


def predict_one(row_features):
    """
    Predict class for a single feature vector (already extracted).
    """
    votes = [0] * len(CLASSES)
    for tree in TREES:
        leaf_votes = _traverse_tree(tree, row_features)
        for i, count in enumerate(leaf_votes):
            votes[i] += count
    best_idx = votes.index(max(votes))
    return CLASSES[best_idx]


def predict(row):
    """
    Helper function to make prediction for a given input row.
    Extracts features from the raw row and calls predict_one.
    """
    # Extract the 22 features in the correct order
    features = [row[col] for col in FEATURE_COLS]
    return predict_one(features)


def predict_all(filename):
    """
    Make predictions for the data in filename.
    """
    # Read the file containing the test data
    df = pd.read_csv(filename)

    predictions = []
    for idx, row in df.iterrows():
        pred = predict(row)
        predictions.append(pred)

    return predictions