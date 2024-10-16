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
