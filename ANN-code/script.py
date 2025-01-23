"""
SSH Script
"""

import numpy as np
import os
from tqdm import tqdm

im_dirs = [
    "/vols/lz/tmarley/GEM_ITO/run/im0",
    "/vols/lz/tmarley/GEM_ITO/run/im1/C",
    "/vols/lz/tmarley/GEM_ITO/run/im1/F",
    "/vols/lz/tmarley/GEM_ITO/run/im2",
    "/vols/lz/tmarley/GEM_ITO/run/im3",
    "/vols/lz/tmarley/GEM_ITO/run/im4"
    ]

# im_dirs = ["ANN-code/Data/im0"]

max_x = 0
max_y = 0
biggest_image_x = None
biggest_image_y = None

for dir in im_dirs:
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith(".npy"):

            full_path = os.path.join(dir, filename)
            image = np.load(full_path)

            y, x = image.shape

            if x > max_x:
                max_x = x
                biggest_image_x = full_path
                
            if y > max_y:
                max_y = y
                biggest_image_y = full_path

print(max_x, max_y)
print("the biggest x image",biggest_image_x)
print("the biggest y image",biggest_image_y)
