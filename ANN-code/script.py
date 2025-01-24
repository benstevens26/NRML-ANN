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
    "/vols/lz/tmarley/GEM_ITO/run/im4",
]

# im_dirs = ["ANN-code/Data/im0"]

max_x = 0
max_y = 0
x_dimensions = []
y_dimensions = []


for dir in im_dirs:
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith(".npy"):

            full_path = os.path.join(dir, filename)
            image = np.load(full_path)

            y, x = image.shape

            x_dimensions.append((x, full_path))
            y_dimensions.append((y, full_path))


x_dimensions.sort(key=lambda item: item[0], reverse=True)
y_dimensions.sort(key=lambda item: item[0], reverse=True)

top_x = x_dimensions[:30]
top_y = y_dimensions[:30]

print("Top 30 images with the largest x dimensions:")
for x_size, path in top_x:
    print(f"Size: {x_size}, Filename: {path}")

print("\nTop 30 images with the largest y dimensions:")
for y_size, path in top_y:
    print(f"Size: {y_size}, Filename: {path}")
