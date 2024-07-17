from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from app.shared import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


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

# Split data into training and testing sets (80/20 plit)
X = snell_df[snell_feature_columns]
y = snell_df["pitch_type"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
