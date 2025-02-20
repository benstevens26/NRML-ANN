#!/usr/bin/env python3
"""
SSH SCRIPT
"""

import pandas as pd
import glob

# Define the output filename pattern
output_filename = "features_Ar_CF4_2.csv"
input_pattern = "features_Ar_CF4_2_*.csv"

# Get all matching feature files
feature_files = sorted(glob.glob(input_pattern))

# Check if any files were found
if not feature_files:
    raise FileNotFoundError("No feature files found to concatenate.")

# Load and concatenate all feature files
df_combined = pd.concat((pd.read_csv(f) for f in feature_files), ignore_index=True)

# Save to final CSV
df_combined.to_csv(output_filename, index=False)

print(f"Concatenated {len(feature_files)} files and saved as {output_filename}")
