import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = "NewYork.csv"
data = pd.read_csv(file_path)

# Filter for a specific location
location_filter = "Manhattan"
data = data[data["Location"] == location_filter]

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Aggregate crime incidents by date
data = data.groupby("Date")["Crime_Incidents"].sum().reset_index()

# Set Date as the index and ensure daily frequency
data.set_index("Date", inplace=True)
data = data.asfreq('D')

# Fill missing values (if any)
data["Crime_Incidents"].fillna(method='ffill', inplace=True)

# Plot the original data
plt.figure(figsize=(10, 5))
plt.plot(data, label="Original Data")
plt.title(f"Crime Incidents Over Time ({location_filter})")
plt.xlabel("Date")
plt.ylabel("Crime Incidents")
plt.legend()
plt.show()

# Fit ARIMA model
model = ARIMA(data["Crime_Incidents"], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1], periods=31, freq="D")[1:]
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(data, label="Original Data")
plt.plot(forecast_series, label="Forecast", color="red")
plt.title(f"ARIMA Forecast of Crime Incidents ({location_filter})")
plt.xlabel("Date")
plt.ylabel("Crime Incidents")
plt.legend()
plt.show()
