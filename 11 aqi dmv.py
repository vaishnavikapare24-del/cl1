#Import Libraries 

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pathlib import Path

warnings.filterwarnings("ignore")


#Load Dataset

DATA_PATH = Path("./datasets/data.csv")

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH, encoding="cp1252")
    print("Loaded original dataset:", DATA_PATH)
else:
    rng = np.random.default_rng(21)
    n = 400

    states = [
        "Maharashtra", "Delhi", "West Bengal",
        "Gujarat", "Kerala", "Rajasthan"
    ]

    data = pd.DataFrame({
        "stn_code": rng.choice(["SAMP", "ST01", "ST02"], n),
        "sampling_date": pd.date_range(
            "2018-01-01", periods=n, freq="D"
        ).strftime("%d-%m-%y"),
        "state": rng.choice(states, n),
        "location": rng.choice(
            ["Pune", "Mumbai", "Delhi", "Kolkata", "Ahmedabad", "Jaipur"], n
        ),
        "agency": rng.choice(
            ["State Pollution Control Board", "Municipal Agency", None], n
        ),
        "type": rng.choice(
            ["Residential", "Industrial", "Sensitive Area", None], n
        ),
        "so2": rng.normal(12, 5, n).clip(0),
        "no2": rng.normal(28, 10, n).clip(0),
        "rspm": rng.normal(100, 35, n).clip(0),
        "spm": rng.normal(180, 50, n).clip(0),
        "pm2_5": rng.normal(60, 20, n).clip(0)
    })

    for col in ["so2", "no2", "rspm", "spm", "pm2_5"]:
        data.loc[
            rng.choice(n, size=20, replace=False), col
        ] = np.nan


# Display Dataset Shape
print("Dataset Shape:", data.shape)

#Display First Five Records
print("First Five Records:")
print(data.head().to_string(index=False))

#Dataset Information
print("Dataset Information:")
data.info()

#Convert Area Types
type_map = {
    "Residential": "R",
    "Residential and others": "RO",
    "Residential, Rural and other Areas": "RRO",
    "Industrial Area": "I",
    "Industrial Areas": "I",
    "Industrial": "I",
    "Sensitive Area": "S",
    "Sensitive Areas": "S",
    "Sensitive": "S"
}

if "type" in data.columns:
    data["type"] = data["type"].replace(type_map).fillna("RRO")


#Convert Pollutant Columns to Numeric
VALUE_COLS = [
    c for c in ["so2", "no2", "rspm", "spm", "pm2_5"]
    if c in data.columns
]

for col in VALUE_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")


#Handle Missing Values Using Mean Imputation
if VALUE_COLS:
    imputer = SimpleImputer(strategy="mean")
    data[VALUE_COLS] = imputer.fit_transform(data[VALUE_COLS])

#Check Missing Values
print("Missing Values After Pollutant Imputation:")
print(data.isnull().sum().to_string())

#Visualize Highest SO2, NO2 and RSPM by State
for indicator in ["so2", "no2", "rspm"]:
    if indicator in data.columns and "state" in data.columns:
        highest = (
            data.groupby("state", dropna=False)[indicator]
            .max()
            .sort_values(ascending=False)
            .head(10)
        )

        plt.figure(figsize=(9, 4))
        highest.plot(kind="bar")
        plt.title(
            f"Top 10 States by Highest Recorded {indicator.upper()}"
        )
        plt.xlabel("State")
        plt.ylabel(indicator.upper())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


#Average Pollutant Levels by Area Type
if "type" in data.columns and VALUE_COLS:
    type_avg = data.groupby("type")[VALUE_COLS].mean(numeric_only=True)

    print("Average Pollutant Levels by Area Type:")
    print(type_avg.to_string())


#Plot Average Pollutant Levels
type_avg.plot(kind="bar", figsize=(9, 5))

plt.title("Average Pollutant Levels by Area Type")
plt.xlabel("Area Type")
plt.ylabel("Average Pollutant Level")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#Completion Message

print("Practical 5 executed successfully.")