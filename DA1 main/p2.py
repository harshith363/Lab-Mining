import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = r'C:\Users\harsh\OneDrive\Documents\VIT\FALLSEM 24-25\CBS3007\DA1 main\oil-prices.csv'
data = pd.read_csv(file_path)

# Strip any leading or trailing spaces from column names
data.columns = data.columns.str.strip()

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

print("Harshith 21BBS0163")
# Line Plot with Moving Average
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Motor Gasoline Price ($/gallon) Real'],
         label='Monthly Price', color='blue')
plt.plot(data.index, data['Motor Gasoline Price ($/gallon) Real'].rolling(
    window=12).mean(), label='12-Month Moving Average', color='red', linestyle='--')
plt.title('Motor Gasoline Prices with 12-Month Moving Average')
plt.xlabel('Date')
plt.ylabel('Price ($/gallon)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Chart of Yearly Average Prices
plt.figure(figsize=(14, 7))
yearly_avg = data['Motor Gasoline Price ($/gallon) Real'].resample('Y').mean()
yearly_avg.plot(kind='bar', color='orange')
plt.title('Average Annual Motor Gasoline Prices')
plt.xlabel('Year')
plt.ylabel('Average Price ($/gallon)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Price Trends Heatmap
monthly_avg = data['Motor Gasoline Price ($/gallon) Real'].resample('M').mean()
monthly_avg = monthly_avg.to_frame().reset_index()
monthly_avg['Month'] = monthly_avg['Date'].dt.month
monthly_avg['Year'] = monthly_avg['Date'].dt.year
pivot_table = monthly_avg.pivot(
    'Month', 'Year', 'Motor Gasoline Price ($/gallon) Real')

plt.figure(figsize=(14, 7))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".2f")
plt.title('Monthly Average Motor Gasoline Prices Heatmap')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

# Histogram of Price Distribution
plt.figure(figsize=(14, 7))
plt.hist(data['Motor Gasoline Price ($/gallon) Real'],
         bins=20, color='green', edgecolor='black')
plt.title('Distribution of Motor Gasoline Prices')
plt.xlabel('Price ($/gallon)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Rolling Statistics Plot
rolling_mean = data['Motor Gasoline Price ($/gallon) Real'].rolling(
    window=12).mean()
rolling_std = data['Motor Gasoline Price ($/gallon) Real'].rolling(
    window=12).std()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Motor Gasoline Price ($/gallon) Real'],
         label='Monthly Price', color='blue')
plt.plot(rolling_mean.index, rolling_mean,
         label='12-Month Rolling Mean', color='red')
plt.plot(rolling_std.index, rolling_std,
         label='12-Month Rolling Std Dev', color='green')
plt.title('Price Trend Analysis with Rolling Statistics')
plt.xlabel('Date')
plt.ylabel('Price ($/gallon)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
