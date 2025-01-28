"""
SSH SCRIPT
"""

import os
import sys
from tqdm import tqdm
import numpy as np
from image_preprocessing import uncropped_check, dim_check, zero_edges
from data_methods import create_file_paths

# image directories
im_dirs = [
    "/vols/lz/tmarley/GEM_ITO/run/im0/C",
    "/vols/lz/tmarley/GEM_ITO/run/im0/F",
    "/vols/lz/tmarley/GEM_ITO/run/im1/C",
    "/vols/lz/tmarley/GEM_ITO/run/im1/F",
    "/vols/lz/tmarley/GEM_ITO/run/im2/C",
    "/vols/lz/tmarley/GEM_ITO/run/im2/F",
    "/vols/lz/tmarley/GEM_ITO/run/im3/C",
    "/vols/lz/tmarley/GEM_ITO/run/im3/F",
    "/vols/lz/tmarley/GEM_ITO/run/im4/C",
    "/vols/lz/tmarley/GEM_ITO/run/im4/F",
]


file_paths = create_file_paths(im_dirs)

min_dim_errors = np.loadtxt("../notebooks/min_dim_list_true.csv", delimiter=",", dtype=str)
uncropped_errors = np.loadtxt("../notebooks/uncropped_list_true.csv", delimiter=",", dtype=str)

# remove the file paths that are in the error lists

print("Number of file paths before removal: ", len(file_paths))
print("Number of min dim errors: ", len(min_dim_errors))
print("Number of uncropped errors: ", len(uncropped_errors))

file_paths = [path for path in file_paths if path not in min_dim_errors and path not in uncropped_errors]

print("Number of file paths after removal: ", len(file_paths))

# now find the max_y and max_x! and save the file path to this image.

max_x = 0
max_y = 0

for path in tqdm(file_paths):


    image = np.load(path)
			
    y, x = image.shape	
				
    if x > max_x:
        max_x = x
        max_x_path = path
    
    if y > max_y:
        max_y = y
        max_y_path = path


print(max_x, max_y)

print("max x file", max_x_path)
print("max y file", max_y_path)