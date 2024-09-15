import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv('part1.csv')
print("Harshith 21BBS0163")

#Question 1
print("First Question")
print("Dimensions:", df.shape)

print("Structure:", df.info())

print("Attribute Names:", df.columns.tolist())

print("Attribute Values:", df.values)


# Question 2
print("Question 2")
# (A) First 5 Records
print(df.head())

# (B) Last 5 Records
print(df.tail())

# (C) Name, Designation, Salary of First 10 Records
print(df[['Name', 'Designation', 'Salary']].head(10))

# (D) Name of All Records
print(df['Name'])

# (E) All Records
print(df)


# Question 3
# a) Mean, Median, Mode of the variables
print("Mean:\n", df.mean())
print("Median:\n", df.median())
print("Mode:\n", df.mode())

# b) Variance and Covariance
print("Variance:\n", df.var())
print("Covariance:\n", df.cov())

# c) Correlation of Salary to Experience
print("Correlation (Salary to Experience):\n",
      df['Salary'].corr(df['Experience']))

# # Question 4
# A) Pie chart on Designation
df['Designation'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Pie Chart of Designation")
plt.show()

# B) Histogram of Salary
df['Salary'].plot(kind='hist', bins=10)
plt.title("Histogram of Salary")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# C) Scatter plot of Salary to Experience
plt.scatter(df['Experience'], df['Salary'])
plt.title("Scatter Plot of Salary to Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
