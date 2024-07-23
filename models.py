from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from app.shared import df
import pandas as pd
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

# Decision Tree Model
dtree_model = DecisionTreeClassifier()

# KNN
knn_model = KNeighborsClassifier(n_neighbors=10)

# List of models and names for df
models = {"Decision Tree": dtree_model, "K Nearest Neighbors": knn_model}


# Make classifications
X = snell_df[snell_feature_columns]  # Feature data
y = snell_df["pitch_type"]  # Target variable

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
        "recall": precision,
        "f1": f1,
    }
    results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

# Function to extract model metrics
def get_model_metric(model_name, metric):
    if model_name in results_df["model"].values:
        return results_df.loc[results_df["model"] == model_name, metric].values[0]
    else:
        return None

print(np.mean(get_model_metric("Decision Tree", "accuracy")))

"""
# Zero rule baseline
most_freq = y_train.mode()
zero_rule_predictions = [most_freq] * len(y_test)
zero_rule_accuracy = accuracy_score(y_test, zero_rule_predictions)
print(f"Zero Rule Baseline Accuracy: {zero_rule_accuracy}")

# Random guessing baseline
class_probabilities = y_train.value_counts(normalize=True)
random_predictions = np.random.choice(
    class_probabilities.index, size=len(y_test), p=class_probabilities.values
)
random_guess_accuracy = accuracy_score(y_test, random_predictions)
print(f"Random Guessing Baseline Accuracy: {random_guess_accuracy}")


# Decision tree modeling
dtree_model = DecisionTreeClassifier().fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
dtree_accuracy = accuracy_score(y_test, dtree_predictions)
dtree_cm = confusion_matrix(y_test, dtree_predictions)
# print(dtree_cm)
print(f"Decision tree accuracy = {dtree_accuracy}")
"""