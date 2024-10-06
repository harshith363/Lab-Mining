import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file

df = pd.read_csv("demat_account_counts.csv")

# Convert Month to a datetime format
df['Month'] = pd.to_datetime(df['Month'])

# Sort data by Month
df = df.sort_values('Month')

# Create a numerical representation of Month for regression
df['Month_Num'] = (df['Month'] - df['Month'].min()).dt.days // 30  # Approximate months

# Features and target
X = df[['Month_Num']]
y = df['Count of DEMAT Accounts']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict for January 2025
# January 2025 is 61 months from the start month (May 2019)
jan_2025_num = (pd.to_datetime('2025-01-01') - df['Month'].min()).days // 30
predicted_count = model.predict([[jan_2025_num]])

print(f"Predicted count of DEMAT accounts for January 2025: {int(predicted_count[0])}")

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(df['Month'], df['Count of DEMAT Accounts'], color='blue', label='Actual Data')
plt.plot(df['Month'], model.predict(X), color='red', label='Regression Line')
plt.scatter(pd.to_datetime('2025-01-01'), predicted_count, color='green', label='Prediction for Jan 2025', s=100)
plt.title("DEMAT Accounts Count Prediction")
plt.xlabel("Month")
plt.ylabel("Count of DEMAT Accounts")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
