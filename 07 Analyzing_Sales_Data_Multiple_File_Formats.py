
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create folder for sample data
DATA_DIR = Path("dmv_sample_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Create sample sales dataset
sales = pd.DataFrame({
    "ORDERNUMBER": [101, 102, 103, 104, 105, 106],
    "QUANTITYORDERED": [20, 35, 15, 40, 25, 18],
    "PRICEEACH": [100.0, 85.5, 120.0, 75.0, 95.0, 110.0],
    "SALES": [2000.0, 2992.5, 1800.0, 3000.0, 2375.0, 1980.0],
    "ORDERDATE": pd.to_datetime([
        "2024-01-05", "2024-01-12", "2024-02-02",
        "2024-02-15", "2024-03-01", "2024-03-20"
    ]),
    "STATUS": [
        "Shipped", "Shipped", "Resolved",
        "Shipped", "On Hold", "Shipped"
    ],
    "PRODUCTLINE": [
        "Classic Cars", "Motorcycles", "Planes",
        "Classic Cars", "Ships", "Trains"
    ]
})

# 2. Save data in CSV, Excel and JSON formats
sales.to_csv(DATA_DIR / "sales.csv", index=False)
sales.to_excel(DATA_DIR / "sales.xlsx", index=False)
sales.to_json(
    DATA_DIR / "sales.json",
    orient="records",
    date_format="iso"
)

print("CSV, Excel and JSON files created in:", DATA_DIR.resolve())
print("\nOriginal Sales Dataset:")
print(sales.head().to_string(index=False))

# 3. Read data from all three file formats
csv_df = pd.read_csv(
    DATA_DIR / "sales.csv",
    parse_dates=["ORDERDATE"]
)

excel_df = pd.read_excel(
    DATA_DIR / "sales.xlsx",
    parse_dates=["ORDERDATE"]
)

json_df = pd.read_json(DATA_DIR / "sales.json")
json_df["ORDERDATE"] = pd.to_datetime(json_df["ORDERDATE"])

print("\nDataset Shapes:")
print("CSV shape:", csv_df.shape)
print("Excel shape:", excel_df.shape)
print("JSON shape:", json_df.shape)

print("\nCSV Dataset Preview:")
print(csv_df.head().to_string(index=False))

# 4. Inspect and clean the data
print("\nDataset Information:")
csv_df.info()

print("\nMissing Values:")
print(csv_df.isnull().sum())

print("\nDuplicate Rows:", csv_df.duplicated().sum())

csv_df = csv_df.drop_duplicates()
csv_df = csv_df.dropna(subset=["SALES"])

print("\nSummary Statistics:")
print(csv_df.describe(include="all").to_string())

# 5. Combine records and remove duplicates
unified_data = pd.concat(
    [csv_df, excel_df, json_df],
    ignore_index=True
)

unified_data = unified_data.drop_duplicates()

total_sales = unified_data["SALES"].sum()

print(f"\nTotal Sales Across Unique Records: {total_sales:.2f}")

# 6. Calculate sales by product line
product_sales = unified_data.groupby(
    "PRODUCTLINE", as_index=False
)["SALES"].sum()

print("\nSales by Product Line:")
print(product_sales.to_string(index=False))

# 7. Visualize sales by product line
product_sales.plot(
    kind="bar",
    x="PRODUCTLINE",
    y="SALES",
    legend=False,
    figsize=(8, 4),
    color="steelblue"
)

plt.title("Sales by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Sales")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
