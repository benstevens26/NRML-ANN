"""
SSH SCRIPT
"""

import numpy as np
import os
from tqdm import tqdm

# Directories to check
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

# Lists to store suspicious filenames
suspicious_y_images = []
suspicious_x_images = []

# Loop through directories
for dir in im_dirs:
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith(".npy"):
            full_path = os.path.join(dir, filename)
            image = np.load(full_path)

            # Get image dimensions
            y_length, x_length = image.shape

            # Find the highest (max y) and widest (max x) non-zero points
            nonzero_y_coords, nonzero_x_coords = np.nonzero(image)
            max_nonzero_y = nonzero_y_coords.max() if nonzero_y_coords.size > 0 else -1
            max_nonzero_x = nonzero_x_coords.max() if nonzero_x_coords.size > 0 else -1

            # Check cropping criteria for y-axis
            if max_nonzero_y < 0.8 * y_length:
                suspicious_y_images.append(full_path)

            # Check cropping criteria for x-axis
            if max_nonzero_x < 0.8 * x_length:
                suspicious_x_images.append(full_path)

# Save the results to files
with open("suspicious_y_images.txt", "w") as f_y:
    for item in suspicious_y_images:
        f_y.write(f"{item}\n")

with open("suspicious_x_images.txt", "w") as f_x:
    for item in suspicious_x_images:
        f_x.write(f"{item}\n")

print("Suspicious y-axis images saved to 'suspicious_y_images.txt'")
print("Suspicious x-axis images saved to 'suspicious_x_images.txt'")
