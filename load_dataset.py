"""
Step 1 of the ML pipeline: Load and combine all CSI CSV files.

This script:
1. Finds all CSV files matching csi_*.csv in the current folder
2. Loads them into one combined pandas DataFrame
3. Parses the csi_values column (currently stored as a string like "[1,2,3,...]")
   back into actual numeric lists
4. Prints summary statistics so you can see how much data you have per class

Run this from the folder containing your CSV files:
    python load_dataset.py
"""

import pandas as pd
import numpy as np
import glob
import ast
import os


def load_all_csi_csvs(folder="."):
    csv_files = glob.glob(os.path.join(folder, "csi_*.csv"))

    if not csv_files:
        print("No CSI CSV files found! Make sure you're running this from the")
        print("folder containing your csi_walking_*.csv, csi_falling_*.csv etc. files.")
        return None

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Parse the csi_values string column back into actual lists of numbers
    combined["csi_values"] = combined["csi_values"].apply(ast.literal_eval)

    # Add a column for how many subcarrier values each sample has (sanity check)
    combined["csi_length"] = combined["csi_values"].apply(len)

    return combined


def main():
    df = load_all_csi_csvs()

    if df is None:
        return

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(f"\nTotal samples: {len(df)}")

    print("\nSamples per activity label:")
    print(df["activity_label"].value_counts())

    print("\nCSI array length distribution (should mostly be consistent):")
    print(df["csi_length"].value_counts())

    print("\nRSSI stats:")
    print(df["rssi"].describe())

    print("\nFirst few rows:")
    print(df[["activity_label", "rssi", "csi_length"]].head(10))

    # Save the combined, cleaned dataset for the next pipeline step
    output_file = "combined_dataset.csv"
    df_to_save = df.copy()
    df_to_save["csi_values"] = df_to_save["csi_values"].apply(str)
    df_to_save.to_csv(output_file, index=False)
    print(f"\nCombined dataset saved to: {output_file}")


if __name__ == "__main__":
    main()