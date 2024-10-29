import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from scipy import fftpack as f

folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use

files = os.listdir(folder_path)
carbon_events = [np.load(folder_path + "/" + f) for f in files]

test_event = carbon_events[0]
plt.imshow(test_event)
plt.show()

###################################################################################################
## trying fourier transform stuff
# transformed_event = f.fft2(test_event)
# plt.imshow(abs(transformed_event))
# plt.show()


# trimmed_transformed_event = np.array([np.append(np.zeros(15),i[15:]) for i in transformed_event])


# trimmed_event = f.fft2(trimmed_transformed_event)
##################################################################################################


filtered_event = nd.gaussian_filter(test_event, 5)
plt.imshow(filtered_event)


# %% principal axis analysis on smoothed profiles
import scipy.ndimage as nd
from feature_extraction import *

folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use

events = load_events(folder_path)

plot = False
for i in events:
    i.paxis = extract_axis(i.image, plot=plot)
    i.smoothed = nd.gaussian_filter(i.image, 5)
    i.smoothed_paxis = extract_axis(i.smoothed, plot=plot)


# %% Total energy conserved
for event in events:
    original_intensity_sum = np.sum(event.image)
    smoothed_intensity_sum = np.sum(event.smoothed)
    event.ratio = smoothed_intensity_sum / original_intensity_sum

plt.plot([i.ratio for i in events], "x")
plt.ylim(0, 1)
plt.show()
# %% Energy to total intensity mapping
intensity_energy_ratios = []
for event in events:
    intensity_energy_ratios.append(np.sum(event.image) / event.energy)

plt.plot(intensity_energy_ratios, "x")
plt.ylabel("Total intensity / energy")
plt.xlabel("event number")
plt.show()
# %%
