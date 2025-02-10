"""
SSH SCRIPT
"""

import os
import sys
from tqdm import tqdm
import numpy as np
from image_preprocessing import uncropped_check, dim_check, zero_edges

# image directories
im_dirs = [
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF40/C",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF40/F",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF40/Ar",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF41/C",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF41/F",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF41/Ar",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF42/C",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF42/F",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF42/Ar",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF43/C",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF43/F",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF43/Ar",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF44/C",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF44/F",
"/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF44/Ar"
]

# check all images in the directory (end in .npy)
min_dim_error = []
uncropped_error = []

# ssearch criteria
min_dim = 10
max_comparison_search_fraction = 0.90
area_comparison_search_fraction = 0.63

# max size
max_x = 0
max_y = 0

for im_dir in im_dirs:
    for file in tqdm(os.listdir(im_dir)):
        if file.endswith(".npy"):
            im_file = os.path.join(im_dir, file)
            image = np.load(im_file)

            if dim_check(image, min_dim):
                min_dim_error.append(im_file)
                continue

            if uncropped_check(zero_edges(image, 2), max_comparison_search_fraction, method="max_comparison"):
                uncropped_error.append(im_file)
                continue

            if uncropped_check(
                zero_edges(image, 2), area_comparison_search_fraction, method="area_comparison"):
                uncropped_error.append(im_file)
                continue

            # no errors found - can check max size
            y, x = image.shape	
				
            if x > max_x:
                max_x = x
                max_x_path = file
        
            if y > max_y:
                max_y = y
                max_y_path = file



print(
    "Number of images that violate the minimum dimension criteria: ", len(min_dim_error)
)
print("Number of images that violate the uncropped criteria: ", len(uncropped_error))

print("Max x: ", max_x, "Path: ", max_x_path)
print("Max y: ", max_y, "Path: ", max_y_path)

# save the min_dim_error and uncropped_error lists as csv files
np.savetxt("min_dim_error_Ar_CF4.csv", min_dim_error, delimiter=",", fmt="%s")
np.savetxt("uncropped_error_Ar_CF4.csv", uncropped_error, delimiter=",", fmt="%s")
