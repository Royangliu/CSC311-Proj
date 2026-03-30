import matplotlib.pyplot as plt # For plotting
import numpy as np              # Linear algebra library
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report # Metrics for evaluation
from sklearn.metrics import ConfusionMatrixDisplay

# imports to visualize tree
from sklearn import tree as treeViz
import graphviz
import pydotplus
from IPython.display import display

def visualize_tree(model, max_depth=5):
    """
    Generate and return an image representing an Sklearn decision tree.

    Each node in the visualization represents a node in the decision tree.
    In addition, visualization for each node contains:
        - The feature that is split on
        - The entropy (of the outputs `t`) at the node
        - The number of training samples at the node
        - The number of training samples with true/false values
        - The majority class (heart disease or not)
    The colour of the node also shows the majority class and purity

    See here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

    Parameters:
        `model` - An Sklearn decision tree model
        `max_depth` - Max depth of decision tree to be rendered in the notebook.
         This is useful since the tree can get very large if the max_depth is
         set too high and thus making the resulting figure difficult to interpret.
    """
    dot_data = treeViz.export_graphviz(model,
                                       feature_names=feature_names,
                                       max_depth=max_depth,
                                       class_names=["heart_no", "heart_yes"],
                                       filled=True,
                                       rounded=True)
    return display(graphviz.Source(dot_data))

# Load the training data
data_train = pd.read_csv("data/train_norm.csv")
data_valid = pd.read_csv("data/val_norm.csv")
data_test = pd.read_csv("data/test_norm.csv")

# Creating Matrix of features and target vector
data_fets_train = np.stack([
    data_train.iloc[:, 2],  # 1-10 emotion ratings
    data_train.iloc[:, 4],    # 1-5 somber rating
    data_train.iloc[:, 5],    # 1-5 content rating
    data_train.iloc[:, 6],    # 1-5 calm rating
    data_train.iloc[:, 7],    # 1-5 uneasy rating
    data_train.iloc[:, 8],    # number of prominent colors
    data_train.iloc[:, 9],    # number of object caught eye
    data_train.iloc[:, 10],   # dollars
    data_train.iloc[:, 16],   # 16-20 one-hot encoded room categories
    data_train.iloc[:, 17],   # 16-20 one-hot encoded room categories
    data_train.iloc[:, 18],   # 16-20 one-hot encoded room categories
    data_train.iloc[:, 19],   # 16-20 one-hot encoded room categories
    data_train.iloc[:, 20],   # 16-20 one-hot encoded room categories
    data_train.iloc[:, 21],   # 16-20 one-hot encoded room categories
    data_train.iloc[:, 22],   # 21-25 one-hot encoded view categories
    data_train.iloc[:, 23],   # 21-25 one-hot encoded view categories
    data_train.iloc[:, 24],   # 21-25 one-hot encoded view categories
    data_train.iloc[:, 25],   # 21-25 one-hot encoded view categories
    data_train.iloc[:, 26],   # Season fall
    data_train.iloc[:, 27],   # Season spring
    data_train.iloc[:, 28],   # Season summer
    data_train.iloc[:, 29],   # Season winter
], axis=1)

data_fets_valid = np.stack([
    data_valid.iloc[:, 2],  # 1-10 emotion ratings
    data_valid.iloc[:, 4],    # 1-5 somber rating
    data_valid.iloc[:, 5],    # 1-5 content rating
    data_valid.iloc[:, 6],    # 1-5 calm rating
    data_valid.iloc[:, 7],    # 1-5 uneasy rating
    data_valid.iloc[:, 8],    # number of prominent colors
    data_valid.iloc[:, 9],    # number of object caught eye
    data_valid.iloc[:, 10],   # dollars
    data_valid.iloc[:, 16],   # 16-20 one-hot encoded room categories
    data_valid.iloc[:, 17],   # 16-20 one-hot encoded room categories
    data_valid.iloc[:, 18],   # 16-20 one-hot encoded room categories
    data_valid.iloc[:, 19],   # 16-20 one-hot encoded room categories
    data_valid.iloc[:, 20],   # 16-20 one-hot encoded room categories
    data_valid.iloc[:, 21],   # 16-20 one-hot encoded room categories
    data_valid.iloc[:, 22],   # 21-25 one-hot encoded view categories
    data_valid.iloc[:, 23],   # 21-25 one-hot encoded view categories
    data_valid.iloc[:, 24],   # 21-25 one-hot encoded view categories
    data_valid.iloc[:, 25],   # 21-25 one-hot encoded view categories
    data_valid.iloc[:, 26],   # Season fall
    data_valid.iloc[:, 27],   # Season spring
    data_valid.iloc[:, 28],   # Season summer
    data_valid.iloc[:, 29],   # Season winter
], axis=1)

data_fets_test = np.stack([
    data_test.iloc[:, 2],  # 1-10 emotion ratings
    data_test.iloc[:, 4],    # 1-5 somber rating
    data_test.iloc[:, 5],    # 1-5 content rating
    data_test.iloc[:, 6],    # 1-5 calm rating
    data_test.iloc[:, 7],    # 1-5 uneasy rating
    data_test.iloc[:, 8],    # number of prominent colors
    data_test.iloc[:, 9],    # number of object caught eye
    data_test.iloc[:, 10],   # dollars
    data_test.iloc[:, 16],   # 16-20 one-hot encoded room categories
    data_test.iloc[:, 17],   # 16-20 one-hot encoded room categories
    data_test.iloc[:, 18],   # 16-20 one-hot encoded room categories
    data_test.iloc[:, 19],   # 16-20 one-hot encoded room categories
    data_test.iloc[:, 20],   # 16-20 one-hot encoded room categories
    data_test.iloc[:, 21],   # 16-20 one-hot encoded room categories
    data_test.iloc[:, 22],   # 21-25 one-hot encoded view categories
    data_test.iloc[:, 23],   # 21-25 one-hot encoded view categories
    data_test.iloc[:, 24],   # 21-25 one-hot encoded view categories
    data_test.iloc[:, 25],   # 21-25 one-hot encoded view categories
    data_test.iloc[:, 26],   # Season fall
    data_test.iloc[:, 27],   # Season spring
    data_test.iloc[:, 28],   # Season summer
    data_test.iloc[:, 29],   # Season winter
], axis=1)

feature_names = [
    "Emotion Rating",
    "Somber Rating",
    "Content Rating",
    "Calm Rating",
    "Uneasy Rating",
    "Prominent Colors",
    "Objects Caught Eye",
    "Dollars",
    "Room Bathroom",
    "Room Bedroom",
    "Room Dining",
    "Room Living",
    "Room Office",
    "View Yourself",
    "View Coworkers/Classmates",
    "View Family",
    "View Friends",
    "View Strangers",
    "Season Fall",
    "Season Spring",
    "Season Summer",
    "Season Winter"
]

# Setup data sets
X_train = data_fets_train
t_train = data_train.iloc[:, 1]  # Painting
X_valid = data_fets_valid
t_valid = data_valid.iloc[:, 1]  # Painting
X_test = data_fets_test
t_test = data_test.iloc[:, 1]  # Painting










# Setup hyperparameters for decision tree
tune_max_depth = [5]
tune_min_impurity_decrease = [0]
tune_n_estimators = [100]

models = []

# Train and evaluate models for each combination of hyperparameters
for max_depth in tune_max_depth:
    for min_impurity_decrease in tune_min_impurity_decrease:
        for n_estimators in tune_n_estimators:
            print(f"Training Random Forest with max_depth={max_depth}, min_impurity_decrease={min_impurity_decrease}, n_estimators={n_estimators}")
            forest_model = RandomForestClassifier(criterion="entropy", max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, n_estimators=n_estimators)
            forest_model.fit(X_train, t_train)
            train_acc = forest_model.score(X_train, t_train)
            valid_acc = forest_model.score(X_valid, t_valid)
            models.append((forest_model, max_depth, min_impurity_decrease, n_estimators, train_acc, valid_acc))


# Print the training and validation accuracy
for model in models:
    print(
        f"Model with max_depth={model[1]}, min_impurity_decrease={model[2]}, n_estimators={model[3]} "
        f"has training accuracy: {model[4]} and validation accuracy: {model[5]}"
    )










# Accuracies of the forest model with the best validation accuracy
best_model = max(models, key=lambda x: x[5])  # Get the model with the highest validation accuracy
chosen_model = best_model[0]
print("Training Accuracy:", chosen_model.score(X_train, t_train))
print("Validation Accuracy:", chosen_model.score(X_valid, t_valid))
print("Test Accuracy:", chosen_model.score(X_test, t_test))










# Confusion matrix on validation set for the chosen model
y_valid_pred = chosen_model.predict(X_valid)
val_conf_matrix = confusion_matrix(t_valid, y_valid_pred)
print("Validation Confusion Matrix:\n", val_conf_matrix)

results_df = pd.DataFrame(
    {
        "max_depth": [model[1] for model in models],
        "min_impurity_decrease": [model[2] for model in models],
        "n_estimators": [model[3] for model in models],
        "train_accuracy": [model[4] for model in models],
        "val_accuracy": [model[5] for model in models],
    }
)












# Plot validation accuracy vs hyperparameters
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=val_conf_matrix, display_labels=chosen_model.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Validation Confusion Matrix (chosen_model)")
plt.tight_layout()
plt.savefig(plots_dir / "validation_confusion_matrix.png", dpi=300)
plt.show()

max_depth_curve = (
    results_df.groupby("max_depth", as_index=False)["val_accuracy"].mean().sort_values("max_depth")
)
plt.figure()
plt.title("Validation Accuracy vs Max Depth")
plt.xlabel("max_depth")
plt.ylabel("Validation Accuracy")
plt.plot(max_depth_curve["max_depth"], max_depth_curve["val_accuracy"], marker="o")
plt.tight_layout()
plt.savefig(plots_dir / "val_accuracy_vs_max_depth.png", dpi=300)
plt.show()

min_impurity_curve = (
    results_df.groupby("min_impurity_decrease", as_index=False)["val_accuracy"]
    .mean()
    .sort_values("min_impurity_decrease")
)
plt.figure()
plt.title("Validation Accuracy vs Min Impurity Decrease")
plt.xlabel("min_impurity_decrease")
plt.ylabel("Validation Accuracy")
plt.plot(min_impurity_curve["min_impurity_decrease"], min_impurity_curve["val_accuracy"], marker="o")
plt.xscale("log")
plt.tight_layout()
plt.savefig(plots_dir / "val_accuracy_vs_min_impurity_decrease.png", dpi=300)
plt.show()

n_estimators_curve = (
    results_df.groupby("n_estimators", as_index=False)["val_accuracy"].mean().sort_values("n_estimators")
)
plt.figure()
plt.title("Validation Accuracy vs Number of Estimators")
plt.xlabel("n_estimators")
plt.ylabel("Validation Accuracy")
plt.plot(n_estimators_curve["n_estimators"], n_estimators_curve["val_accuracy"], marker="o")
plt.tight_layout()
plt.savefig(plots_dir / "val_accuracy_vs_n_estimators.png", dpi=300)
plt.show()


# Save the decision forest to a json file
import json

forest = []

for estimator in chosen_model.estimators_:
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