import pandas as pd
import os
from glob import glob

DATASET_FOLDER = "datasets"

csv_files = glob(os.path.join(DATASET_FOLDER, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

all_data = []

for file in csv_files:
    print(f"Reading {os.path.basename(file)}")
    df = pd.read_csv(file)
    all_data.append(df)

merged_df = pd.concat(all_data, ignore_index=True)

print("\nMerged Dataset Shape:")
print(merged_df.shape)

print("\nColumns:")
print(merged_df.columns)

merged_df.to_csv("merged_air_quality.csv", index=False)

print("\nMerged dataset saved as merged_air_quality.csv")