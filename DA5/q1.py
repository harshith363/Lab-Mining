import pandas as pd 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA 
import matplotlib.pyplot as plt 

market_df = pd.read_excel("TAMO Historical Data.xlsx") 
market_df["SMA_Short"] = market_df["Price"].rolling(window=50).mean() 
market_df["SMA_Medium"] = market_df["Price"].rolling(window=200).mean() 
market_df["SMA_Long"] = market_df["Price"].rolling(window=365).mean() 
market_df["SMA_Extended"] = market_df["Price"].rolling(window=500).mean() 

time_series = ARIMA(market_df["Price"], order=(5, 1, 0))   
fitted_ts = time_series.fit() 
predictions = fitted_ts.forecast(steps=30) 
print("Predicted Prices for Next 30 Days:") 
print(predictions) 

plt.figure(figsize=(14, 7)) 
plt.plot(market_df["Date"], market_df["Price"], label="Asset Prices") 
plt.plot(market_df["Date"], market_df["SMA_Short"], label="50-day Simple Moving Average", linestyle="--") 
plt.plot(market_df["Date"], market_df["SMA_Medium"], label="200-day Simple Moving Average", linestyle="--") 
plt.legend() 
plt.title("Asset Performance Analysis") 
plt.xlabel("Timeline") 
plt.ylabel("Price") 
plt.show()