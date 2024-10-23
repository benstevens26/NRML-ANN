"""
test module
"""
import random
from feature_extraction import *

folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use
events = load_events(folder_path)

# sample some events for analysis
events = random.sample(events, 10)

# test event
test_event = events[0]

test_event.plot_intensity_profile(num_segments=50)





