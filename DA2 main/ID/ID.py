import pandas as pd
import numpy as np
from math import log2

# Load dataset
data = pd.read_csv('data.csv')

# Calculate entropy


def entropy(y):
    total = len(y)
    counts = y.value_counts()
    probs = counts / total
    # Added a small value to avoid log2(0)
    entropy_value = -np.sum(probs * np.log2(probs + 1e-9))
    return entropy_value

# Calculate information gain


def information_gain(X_column, y):
    total_entropy = entropy(y)
    values, counts = np.unique(X_column, return_counts=True)
    weighted_entropy = np.sum(
        (counts[i] / np.sum(counts)) * entropy(y[X_column == values[i]]) for i in range(len(values)))
    gain = total_entropy - weighted_entropy
    return gain

# Build the decision tree


def id3(X, y, attributes):
    # Convert y to pandas Series to use .unique()
    y = pd.Series(y)

    # If all target values are the same, return that value
    if len(y.unique()) == 1:
        return y.iloc[0]

    # If no more attributes, return the most common target value
    if len(attributes) == 0:
        return y.mode()[0]

    # Choose the best attribute
    best_attribute = max(
        attributes, key=lambda attr: information_gain(X[attr], y))
    tree = {best_attribute: {}}

    for value in X[best_attribute].unique():
        sub_X = X[X[best_attribute] == value]
        sub_y = y[X[best_attribute] == value]
        subtree = id3(
            sub_X, sub_y, [attr for attr in attributes if attr != best_attribute])
        tree[best_attribute][value] = subtree

    return tree


# Prepare data
X = data[['Length', 'Numberof_Bends', 'Trafficvolume']]
y = data['AccidentRisk']
attributes = ['Length', 'Numberof_Bends', 'Trafficvolume']

# Encode categorical target variable
y_encoded, y_mapping = pd.factorize(y)

# Encode categorical features (if needed)
X_encoded = X.copy()
for col in X_encoded.columns:
    X_encoded[col], _ = pd.factorize(X_encoded[col])

# Build the decision tree
decision_tree = id3(X_encoded, y_encoded, attributes)

# Print the decision tree
print("Decision Tree:")
print(decision_tree)
