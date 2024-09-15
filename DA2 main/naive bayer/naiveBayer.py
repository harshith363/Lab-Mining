import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("21BBS0163 HARSHITH KUMAR")

df = pd.read_csv('diet.csv')

print(df.head())

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Physical Activity Level'] = le.fit_transform(df['Physical Activity Level'])
df['Dietary Habit'] = le.fit_transform(df['Dietary Habit'])
df['Strict Diet'] = le.fit_transform(df['Strict Diet'])

X = df[['Gender', 'Age', 'Weight', 'Height', 'BMI',
        'Physical Activity Level']]
y = df['Strict Diet']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
