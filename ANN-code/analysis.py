import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from convert_sim_ims import *
from feature_extraction import *

folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use
events = load_events(folder_path)


m_dark_list = [np.load(f) for f in glob.glob("Data/darks/master_dark_*.npy")]
m_dark = m_dark_list[
    1
]  # This is only for while I don't care about the others. I cba to implement ordering them by the numbers at the end of the file names
example_dark = np.load("Data/darks/quest_std_dark_1.npy")[
    np.random.randint(0, 199)
]  # This is an array of 200 dark images. I believe the other 9 are the same as well

dark_sample = get_dark_sample(
    m_dark, [len(events[0].image[0]), len(events[0].image)], example_dark
)
plt.imshow(dark_sample)

events[0].noisy_image = convert_im(events[0].image, dark_sample)
plt.imshow(events[0].noisy_image)
