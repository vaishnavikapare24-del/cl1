import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

USE_LIVE_API = False

if USE_LIVE_API:
    import requests

    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    if not API_KEY:
        raise ValueError("Please set the OPENWEATHER_API_KEY environment variable.")

    city = "Pune"
    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    weather_data = response.json()

    records = []

    for item in weather_data["list"]:
        records.append({
            "timestamp": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"]["speed"],
            "weather_description": item["weather"][0]["description"]
        })

    df = pd.DataFrame(records)

else:
    np.random.seed(42)

    timestamps = pd.date_range(
        start="2024-10-25",
        periods=48,
        freq="3h"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.random.uniform(20, 35, 48),
        "humidity": np.random.uniform(40, 90, 48),
        "wind_speed": np.random.uniform(1, 15, 48),
        "weather_description": np.random.choice(
            ["clear sky", "cloudy", "rain", "few clouds"],
            48
        )
    })

print("\nFirst Five Records:")
print(df.head().to_string(index=False))

print("\nDataset Information:")
df.info()

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df.dropna(subset=["timestamp"], inplace=True)

numeric_columns = ["temperature", "humidity", "wind_speed"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].fillna(df[column].mean())

df.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df[numeric_columns].describe())

df["date"] = df["timestamp"].dt.date

daily_mean = df.groupby("date")[numeric_columns].mean().reset_index()

print("\nDaily Average Weather:")
print(daily_mean.head().to_string(index=False))

plt.figure(figsize=(10, 5))
plt.plot(daily_mean["date"], daily_mean["temperature"], marker="o")
plt.title("Daily Average Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(daily_mean["date"].astype(str), daily_mean["humidity"])
plt.title("Daily Average Humidity")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="temperature", y="wind_speed")
plt.title("Temperature vs Wind Speed")
plt.xlabel("Temperature (°C)")
plt.ylabel("Wind Speed (m/s)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
correlation = df[numeric_columns].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Weather Data Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nPractical 2 executed successfully.")