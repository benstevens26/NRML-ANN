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


# check all images in the directory (end in .npy)
min_dim_error = []
uncropped_error = []

# ssearch criteria
min_dim = 20

for im_dir in im_dirs:
    for file in tqdm(os.listdir(im_dir)):
        if file.endswith(".npy"):
            im_file = os.path.join(im_dir, file)
            image = np.load(im_file)

            if dim_check(image, min_dim):
                min_dim_error.append(im_file)
                continue

            if uncropped_check(image, search_fraction=0.8, method='max_comparison'):
                uncropped_error.append(im_file)
                continue

            if uncropped_check(zero_edges(image, 5), search_fraction=0.6, method='area_comparison'):
                uncropped_error.append(im_file)
                

print("Number of images that violate the minimum dimension criteria: ", len(min_dim_error))
print("Number of images that violate the uncropped criteria: ", len(uncropped_error))

# save the min_dim_error and uncropped_error lists as csv files
np.savetxt("min_dim_error.csv", min_dim_error, delimiter=",", fmt="%s")
np.savetxt("uncropped_error.csv", uncropped_error, delimiter=",", fmt="%s")

