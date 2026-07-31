import pandas as pd
import numpy as np

# Load merged dataset
df = pd.read_csv("merged_air_quality.csv")

print("Original Shape:", df.shape)

# Keep only important columns
columns = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

df = df[columns]

# Fill missing values using median
df = df.fillna(df.median(numeric_only=True))

# ---------- Health Risk Score ----------
risk = (
        0.35 * df["PM2.5"] +
        0.20 * df["PM10"] +
        0.10 * df["NO2"] +
        0.10 * df["SO2"] +
        0.10 * df["CO"] +
        0.05 * df["O3"] -
        0.03 * df["WSPM"] +
        0.02 * df["TEMP"]
)

# Normalize score to 0–100
risk = (risk - risk.min()) / (risk.max() - risk.min()) * 100

df["HealthRisk"] = risk

print(df.head())

print("\nFinal Shape:", df.shape)

df.to_csv("clean_air_quality.csv", index=False)

print("\nSaved clean_air_quality.csv")