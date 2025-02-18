import os
import pandas as pd
from data_methods import get_file_paths, match_files

# Directories for camera and ITO images
base_dirs_cam = [
    "/vols/lz/tmarley/GEM_ITO/run/im0/C", "/vols/lz/tmarley/GEM_ITO/run/im0/F",
    "/vols/lz/tmarley/GEM_ITO/run/im1/C", "/vols/lz/tmarley/GEM_ITO/run/im1/F",
    "/vols/lz/tmarley/GEM_ITO/run/im2/C", "/vols/lz/tmarley/GEM_ITO/run/im2/F",
    "/vols/lz/tmarley/GEM_ITO/run/im3/C", "/vols/lz/tmarley/GEM_ITO/run/im3/F",
    "/vols/lz/tmarley/GEM_ITO/run/im4/C", "/vols/lz/tmarley/GEM_ITO/run/im4/F"
]

base_dirs_ito = [
    "/vols/lz/tmarley/GEM_ITO/run/ito_npy0/C", "/vols/lz/tmarley/GEM_ITO/run/ito_npy0/F",
    "/vols/lz/tmarley/GEM_ITO/run/ito_npy1/C", "/vols/lz/tmarley/GEM_ITO/run/ito_npy1/F",
    "/vols/lz/tmarley/GEM_ITO/run/ito_npy2",
    "/vols/lz/tmarley/GEM_ITO/run/ito_npy3",
    "/vols/lz/tmarley/GEM_ITO/run/ito_npy4"
]

# Generate matched dataframe
df_matched = match_files(base_dirs_cam, base_dirs_ito)

# Save to CSV
df_matched.to_csv("matched_file_paths_CF4.csv", index=False)

print(f"Matched {len(df_matched)} events and saved to 'matched_file_paths_CF4.csv'")
