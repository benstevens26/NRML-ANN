from bb_event import load_events_bb
from bb_event import Event
import numpy as np


def data_summary(events):
    """ """
    pass


events = load_events_bb("ANN-code/Data/im0")

# sample 20 events randomly
events = np.random.choice(events, 20, replace=False)

events[0].plot_image()
print(events[0].name)
print(events[0].species)
