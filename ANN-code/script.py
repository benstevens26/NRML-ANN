"""
SSH Script
"""

import numpy as np
import os
from tqdm import tqdm

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

max_x = 0
max_y = 0
max_x_image = None
max_y_image = None

for dir in im_dirs:
	for filename in tqdm(os.listdir(dir)):
		if filename.endswith(".npy"):

			full_path = os.path.join(dir, filename)
			image = np.load(full_path)
			
			y, x = image.shape	
				
			if x > max_x:
				max_x = x
				max_x_image = full_path
			
			if y > max_y:
				max_y = y
				max_y_image = full_path

print(max_x, max_y)
print(max_x_image)
print(max_y_image)