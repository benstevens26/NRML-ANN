"""
analysis

"""

from event import Event
from event import load_events
import numpy as np
import random
import tqdm

folder_path = "Data/C/300-320keV"
events = load_events(folder_path)

# sample some events for analysis
events = random.sample(events, 10)
test_event = events[0]
test_event.plot_image_with_axis(image_type="raw")
test_event.plot_image_with_axis()
#test_event.plot_intensity_profile(num_segments=250)
