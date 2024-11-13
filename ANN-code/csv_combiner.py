# combine_results.py
import glob
import numpy as np
import pandas as pd

output_csv_prefix = "2x2_binned_features"
# Read all output files
nums = np.arange(20)  # CHANGE TO RIGHT NUMBER
all_files = [
    f"/vols/lz/twatson/ANN/analysis_outputs/{output_csv_prefix}_{str(i)}.csv"
    for i in nums
]
all_data = [pd.read_csv(f) for f in all_files]

# Concatenate and save
final_data = pd.concat(all_data, ignore_index=True)
final_data.to_csv("all_2x2_binned_features.csv", index=False)
