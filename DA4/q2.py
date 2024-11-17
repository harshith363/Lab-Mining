import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('q2.csv')

print("Data Overview:")
print(df.head())

label_encoder = LabelEncoder()
df['Income'] = label_encoder.fit_transform(df['Income'])
df['Criminal_Record'] = label_encoder.fit_transform(df['Criminal Record'])
df['EXP'] = label_encoder.fit_transform(df['EXP'])
df['Loan_Approved'] = label_encoder.fit_transform(df['Loan_Approved'])

X = df[['Income', 'Criminal_Record', 'EXP']]
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

input_data = [[1, 1, 2]]

probabilities = gnb.predict_proba(input_data)

approved_prob = probabilities[0][1]  #
not_approved_prob = probabilities[0][0]

print(f"Probability of loan approval: {approved_prob:.2f}")
print(f"Probability of loan not being approved: {not_approved_prob:.2f}")
