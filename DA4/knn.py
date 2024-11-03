import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random

data = pd.read_csv('q1.csv')


# Create DataFrame
df = pd.DataFrame(data)

# Create a binary classification target based on BMI
df['Height_m'] = df['Height'] / 100  # Convert height to meters
df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
df['Weight_Category'] = (df['BMI'] > 25).astype(
    int)  # 1 if overweight, 0 if normal

# Prepare features for KNN
X = df[['Height', 'Weight', 'Age']]
y = df['Weight_Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
k = 5  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Print results
print("\nDataset Sample:")
print(df.head())
print("\nFeature Statistics:")
print(df[['Height', 'Weight', 'Age']].describe())
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to predict for new data


def predict_weight_category(height, weight, age):
    new_data = np.array([[height, weight, age]])
    new_data_scaled = scaler.transform(new_data)
    prediction = knn.predict(new_data_scaled)
    return "Overweight" if prediction[0] == 1 else "Normal weight"


# Example prediction
example_height = 175
example_weight = 80
example_age = 30
print(f"\nExample Prediction:")
print(
    f"For a person with Height: {example_height}cm, Weight: {example_weight}kg, Age: {example_age}")
print(
    f"Predicted category: {predict_weight_category(example_height, example_weight, example_age)}")
