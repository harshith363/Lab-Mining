import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from ta.trend import SMAIndicator, EMAIndicator

# Load the dataset
print("Harshith Kumar 21BBS0163")
file_path = "NewYork.csv"
data = pd.read_csv(file_path)

# Filter for a specific location
location_filter = "Manhattan"
data = data[data["Location"] == location_filter]

# Convert Date column to datetime and set as index
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Aggregate daily data and ensure frequency
data = data.resample("D").sum()  # Sum up incidents per day if duplicates
data["Crime_Incidents"].fillna(method='ffill', inplace=True)

# Define moving averages
window = 7  # Example window for moving averages
data["SMA"] = data["Crime_Incidents"].rolling(
    window=window).mean()  # Simple Moving Average
data["EMA"] = data["Crime_Incidents"].ewm(
    span=window, adjust=False).mean()  # Exponential Moving Average
data["WMA"] = data["Crime_Incidents"].rolling(window=window).apply(
    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
)  # Weighted Moving Average
data["HMA"] = data["Crime_Incidents"].rolling(window=window).apply(
    lambda x: np.mean(2 * x - x.rolling(window=int(window / 2)).mean())
)  # Hull Moving Average (Approximation)
data["Moving_Average_Crossover"] = np.where(
    data["EMA"] > data["SMA"], 1, 0)  # Crossover Signal

# Plot the Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(data["Crime_Incidents"], label="Original Data", color="blue")
plt.plot(data["SMA"], label="Simple Moving Average (SMA)",
         color="orange", linestyle="--")
plt.plot(data["EMA"], label="Exponential Moving Average (EMA)",
         color="green", linestyle="--")
plt.plot(data["WMA"], label="Weighted Moving Average (WMA)",
         color="red", linestyle="--")
plt.plot(data["HMA"], label="Hull Moving Average (HMA)",
         color="purple", linestyle="--")
plt.title("Moving Averages - Crime Data")
plt.xlabel("Date")
plt.ylabel("Crime Incidents")
plt.legend()
plt.grid()
plt.show()

# ARIMA (Autoregressive Moving Average)
model = ARIMA(data["Crime_Incidents"], order=(1, 0, 1))
model_fit = model.fit()

# Plot ARIMA Residuals
residuals = model_fit.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals, label="ARIMA Residuals", color="red")
plt.title("ARIMA Residuals")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.legend()
plt.grid()
plt.show()

# Moving Average Crossover Strategy Visualization
plt.figure(figsize=(12, 6))
plt.plot(data["Crime_Incidents"], label="Original Data", color="blue")
plt.plot(data["SMA"], label="SMA", color="orange", linestyle="--")
plt.plot(data["EMA"], label="EMA", color="green", linestyle="--")
plt.fill_between(data.index, data["Crime_Incidents"].min(), data["Crime_Incidents"].max(),
                 where=data["Moving_Average_Crossover"] == 1, color='lightgreen', alpha=0.2,
                 label="EMA > SMA (Bullish)")
plt.fill_between(data.index, data["Crime_Incidents"].min(), data["Crime_Incidents"].max(),
                 where=data["Moving_Average_Crossover"] == 0, color='lightcoral', alpha=0.2,
                 label="EMA <= SMA (Bearish)")
plt.title("Moving Average Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Crime Incidents")
plt.legend()
plt.grid()
plt.show()
