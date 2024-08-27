from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from app.shared import df
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Multiclass Classification For Blake Snell Pitch Type
snell_df = df[df["pitcher_name"] == "Snell, Blake"]

# Convert on categorical columns to numeric
on_base_columns = ["on_3b", "on_2b", "on_1b"]
for col in on_base_columns:
    snell_df.loc[:, col] = snell_df[col].notna()

snell_df.loc[:, "stand"] = snell_df["stand"].replace({"L": 0, "R": 1})

# Select feature columns
snell_feature_columns = [
    "stand",
    "balls",
    "strikes",
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning",
    "at_bat_number",
    "pitch_number",
    "bat_score",
    "fld_score",
]

# Create df for model results and statistics
columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
]
results_df = pd.DataFrame(columns=columns)


# Define Models

# Dummy Classifiers for Baseline
zero_rule = DummyClassifier(strategy="most_frequent")
random_rate = DummyClassifier(strategy="stratified")

# Decision Tree Model
dtree_model = DecisionTreeClassifier()

# KNN
knn_model = KNeighborsClassifier(
    n_neighbors=9,
)

# List of models and names for df
models = {
    "Zero Rule": zero_rule,
    "Random Rate": random_rate,
    "Decision Tree": dtree_model,
    "kNN": knn_model,
}


# Make classifications
X = snell_df[snell_feature_columns]  # Feature data
y = snell_df["pitch_type"]  # Target variable
# print(y.dropna().unique().tolist())

# 10-fold cross-validation with 5 repeats
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
# Loop for each model
for model_name, model in models.items():
    # Evaluate models
    accuracy = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    precision = cross_val_score(
        model, X, y, scoring="precision_macro", cv=cv, n_jobs=-1
    )
    recall = cross_val_score(model, X, y, scoring="recall_macro", cv=cv, n_jobs=-1)
    f1 = cross_val_score(model, X, y, scoring="f1_macro", cv=cv, n_jobs=-1)

    # Store model performance metrics
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)


# Function to extract model metrics
def get_model_metric(model_name, metric):
    if model_name in results_df["model"].values:
        return results_df.loc[results_df["model"] == model_name, metric].values[0]
    else:
        return None


colors = ["#da0bde", "#a81f0a", "#0a49c9", "#07ad44"]

# Plot Precision Comparison
precision_list = [
    get_model_metric(model_name, "precision") for model_name in models.keys()
]
fig, ax = plt.subplots()
bp = ax.boxplot(precision_list, patch_artist=True, notch=True)
ax.set_xticklabels(models.keys())
ax.set_ylabel("Precision Score")
ax.set_title("Precision Scores of Different Models")
ax.yaxis.grid(True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
# plt.show()
plt.savefig("./app/plots/precision_score.png", format="png", dpi=1200)

# Plot Recall Comparison
recall_list = [get_model_metric(model_name, "recall") for model_name in models.keys()]
fig, ax = plt.subplots()
bp = ax.boxplot(recall_list, patch_artist=True, notch=True)
ax.set_xticklabels(models.keys())
ax.set_ylabel("Recall")
ax.set_title("Recall of Different Models")
ax.yaxis.grid(True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
# plt.show()
plt.savefig("./app/plots/recall.png", format="png", dpi=1200)

# Plot F1 Comparison
f1_list = [get_model_metric(model_name, "f1") for model_name in models.keys()]
fig, ax = plt.subplots()
bp = ax.boxplot(f1_list, patch_artist=True, notch=True)
ax.set_xticklabels(models.keys())
ax.set_ylabel("F-Score")
ax.set_title("F-Score of Different Models")
ax.yaxis.grid(True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
# plt.show()
plt.savefig("./app/plots/f_score.png", format="png", dpi=1200)

# Plot Accuracy Comparison
accuracy_list = [
    get_model_metric(model_name, "accuracy") for model_name in models.keys()
]
fig, ax = plt.subplots()
bp = ax.boxplot(accuracy_list, patch_artist=True, notch=True)
ax.set_xticklabels(models.keys())
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of Different Models")
ax.yaxis.grid(True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
# plt.show()
plt.savefig("./app/plots/accuracy.png", format="png", dpi=1200)