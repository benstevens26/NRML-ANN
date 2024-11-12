# combine_results.py
import glob

import pandas as pd

output_csv_prefix = "2x2_binned_features"
# Read all output files
all_files = glob.glob(f"{output_csv_prefix}_*.csv")
all_data = [pd.read_csv(f) for f in all_files]

# Concatenate and save
final_data = pd.concat(all_data, ignore_index=True)
final_data.to_csv("final_output.csv", index=False)
