#!/usr/bin/env python3
"""
SSH SCRIPT
"""

import pandas as pd

# List to hold dataframes
dfs = []

# Read in the CSV files
for i in range(15):
    df = pd.read_csv(f'features_Ar_CF4_{i}.csv')
    dfs.append(df)

# Concatenate all dataframes
concatenated_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated dataframe to a new CSV file
concatenated_df.to_csv('features_Ar_CF4.csv', index=False)