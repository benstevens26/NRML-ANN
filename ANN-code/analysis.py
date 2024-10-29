"""
analysis

"""

from event import Event
from event import load_events
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm

folder_path = "Data/C/300-320keV"
events = load_events(folder_path)
events = random.sample(events, 20)

test_event = events[0]

test_event.plot_image()

principal_axis, mean_x, mean_y = test_event.get_principal_axis()
print(test_event.get_recoil_angle(principal_axis=principal_axis))

print(principal_axis)







