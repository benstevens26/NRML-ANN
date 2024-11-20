# combine_results.py
import glob
import numpy as np
import pandas as pd

output_csv_prefix = "2x2_binned_features"
# Read all output files
nums = np.arange(40)  # CHANGE TO RIGHT NUMBER
all_files = [
    f"/vols/lz/twatson/ANN/analysis_outputs/{output_csv_prefix}_{str(i)}.csv"
    for i in nums
]

# Initialize an empty list to store successfully read data
all_data = []

for file in all_files:
    try:
        # Attempt to read the file
        data = pd.read_csv(file)
        all_data.append(data)
    except Exception as e:
        # Skip the file and print an error message
        print(f"Error reading file {file}: {e}")

# Concatenate and save if there is any data
if all_data:
    final_data = pd.concat(all_data, ignore_index=True)
    final_data.to_csv("30_segs_2x2_binned_features.csv", index=False)
else:
    print("No valid data to concatenate.")
