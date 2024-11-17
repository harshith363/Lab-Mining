import pandas as pd
import numpy as np

# Generate a date range for one year
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')

# Simulate crime data
np.random.seed(42)
locations = ["Central", "East", "West", "North", "South"]
crime_data = {
    "Date": np.tile(dates, len(locations)),
    "Location": np.repeat(locations, len(dates)),
    "Crime_Incidents": np.random.randint(10, 30, size=len(dates) * len(locations)),
}

# Create DataFrame
df = pd.DataFrame(crime_data)

# Sample 300 rows, ensuring balanced representation across locations
sampled_df = df.groupby("Location").apply(
    lambda x: x.sample(n=500 // len(locations), random_state=42)
).reset_index(drop=True)

# Save the sampled dataset to CSV
sampled_df.to_csv("metrocity_crime_data.csv", index=False)

# Output a quick preview of the sampled dataset
print(sampled_df.head())
