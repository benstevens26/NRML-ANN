# feature extraction for input into the ANN

import numpy as np
import matplotlib.pyplot as plt
import os

folder_path = os.path.join('..', '..', 'Data/C/300-320keV')
# folder_path = os.path.join( tom path )

files = os.listdir(folder_path)
carbon_events = [np.load(folder_path + "/" + f) for f in files]

event1 = carbon_events[0]

plt.figure()
plt.imshow(event1, cmap='viridis')
plt.colorbar()
plt.show()