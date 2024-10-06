import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Harshith Kumar 21BBS0163")

df = pd.read_csv("student.csv")

X = df[['Attendance', 'Marks (Data Mining)']]

y = np.where(X['Attendance'] < 75, 'Drop',
             np.where(X['Marks (Data Mining)'] < 40, 'Fail', 'Pass'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

all_predictions = knn.predict(scaler.transform(X))

df['Prediction'] = all_predictions

print("\nStudent Classifications:")
for _, row in df.iterrows():
    print(
        f"Roll Number: {row['Roll Number']}, Classification: {row['Prediction']}")


# Plotting
plt.figure(figsize=(12, 6))

# Scatter plot for Attendance vs Marks, colored by Prediction
sns.scatterplot(data=df, x='Attendance', y='Marks (Data Mining)', hue='Prediction',
                palette='Set1', style='Prediction', markers={"Drop": "X", "Fail": "o", "Pass": "s"}, s=100)
plt.title("Student Performance Classification")
plt.xlabel("Attendance (%)")
plt.ylabel("Marks (Data Mining)")
plt.axhline(40, color='red', linestyle='--',
            label='Pass/Fail Threshold (40 Marks)')
plt.axvline(75, color='orange', linestyle='--',
            label='Drop Threshold (75 Attendance)')
plt.legend()
plt.grid()
plt.show()

# Count the number of students in each category
drop_count = (df['Prediction'] == 'Drop').sum()
fail_count = (df['Prediction'] == 'Fail').sum()
pass_count = (df['Prediction'] == 'Pass').sum()

print(f"\nTotal students at risk of dropping out: {drop_count}")
print(f"Total failing students: {fail_count}")
print(f"Total passing students: {pass_count}")
