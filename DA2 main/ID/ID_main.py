import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder

print("21BBS0163 HARSHITH KUMAR")

# Load the dataset
df = pd.read_csv('data.csv')
df.head()

# Encode the target variable
label_encoder = LabelEncoder()
df['AccidentRisk'] = label_encoder.fit_transform(df['AccidentRisk'])

# Function to calculate entropy


def calculate_entropy(data, target_column):
    total_rows = len(data)
    target_values = data[target_column].unique()

    entropy = 0
    for value in target_values:
        value_count = len(data[data[target_column] == value])
        proportion = value_count / total_rows
        entropy -= proportion * math.log2(proportion) if proportion != 0 else 0

    return entropy

# Function to calculate information gain


def calculate_information_gain(data, feature, target_column, entropy_outcome):
    unique_values = data[feature].unique()
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[feature] == value]
        proportion = len(subset) / len(data)
        weighted_entropy += proportion * \
            calculate_entropy(subset, target_column)

    information_gain = entropy_outcome - weighted_entropy
    return information_gain


# Calculate the entropy of the target variable
entropy_outcome = calculate_entropy(df, 'AccidentRisk')

# Calculate and print entropy and information gain for each feature
print("\nEntropy and Information Gain for each feature:")
for column in df.columns[:-1]:
    entropy = calculate_entropy(df, column)
    information_gain = calculate_information_gain(
        df, column, 'AccidentRisk', entropy_outcome)
    print(f"{column} - Entropy: {entropy:.3f}, Information Gain: {information_gain:.3f}")

# Feature selection for the first step in making decision tree
selected_feature = 'Length'  # Example feature

# Ensure the selected feature is numeric
if df[selected_feature].dtype == 'object':
    df[selected_feature] = label_encoder.fit_transform(df[selected_feature])

# Prepare the data for training
X = df[[selected_feature]]
y = df['AccidentRisk']

# Create and train a decision tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=[
          selected_feature], class_names=label_encoder.classes_, filled=True, rounded=True)
plt.show()

# Implement the ID3 algorithm


def id3(data, target_column, features):
    # If all target values are the same, return that value (leaf node)
    if len(data[target_column].unique()) == 1:
        return data[target_column].iloc[0]

    # If no more features, return the most common target value
    if len(features) == 0:
        return data[target_column].mode().iloc[0]

    # Find the feature with the highest information gain
    best_feature = max(features, key=lambda x: calculate_information_gain(
        data, x, target_column, entropy_outcome))

    # Create a new decision tree with the best feature as a node
    tree = {best_feature: {}}

    # Remove the best feature from the remaining features
    features = [f for f in features if f != best_feature]

    # Recursively split the data by the feature values
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, target_column, features)
        tree[best_feature][value] = subtree

    return tree


# List of features (excluding the target column)
features = list(df.columns[:-1])

# Build the decision tree using the ID3 algorithm
decision_tree = id3(df, 'AccidentRisk', features)

# Print the resulting decision tree
print("\nGenerated Decision Tree using ID3 algorithm:")
print(decision_tree)
