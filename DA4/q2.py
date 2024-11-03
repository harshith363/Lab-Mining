import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
df = pd.read_csv('q2.csv')

# Display the first few rows of the dataset
print("Data Overview:")
print(df.head())

# Encode categorical variables
label_encoder = LabelEncoder()
df['Income'] = label_encoder.fit_transform(df['Income'])
df['Criminal_Record'] = label_encoder.fit_transform(df['Criminal Record'])
df['EXP'] = label_encoder.fit_transform(df['EXP'])
df['Loan_Approved'] = label_encoder.fit_transform(df['Loan_Approved'])

# Features and target variable
X = df[['Income', 'Criminal_Record', 'EXP']]
y = df['Loan_Approved']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Given conditions for prediction: Age 30-70, Criminal Record Yes, Experience >5
# Encode input data: Income = 1 (30-70), Criminal Record = 1 (Yes), EXP = 2 (>5)
input_data = [[1, 1, 2]]

# Predict the probability of loan approval
probabilities = gnb.predict_proba(input_data)

# Output probabilities
approved_prob = probabilities[0][1]  # Probability of "Loan Approved" (Yes)
# Probability of "Loan Not Approved" (No)
not_approved_prob = probabilities[0][0]

print(f"Probability of loan approval: {approved_prob:.2f}")
print(f"Probability of loan not being approved: {not_approved_prob:.2f}")
