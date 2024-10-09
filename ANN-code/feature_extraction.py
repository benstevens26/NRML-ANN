# feature extraction for input into the ANN

import numpy as np
import matplotlib.pyplot as plt
import os

folder_path = os.path.join('..', '..', 'Data/C/300-320keV')
files = os.listdir(folder_path)

carbon_events = np.array(files)

print(carbon_events)




