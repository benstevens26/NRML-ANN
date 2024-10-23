"""
feature_extraction.py

This module provides functionality to handle and process image data for input into an Artificial Neural Network (ANN).
It includes classes and methods to load events, process images, and extract relevant features for further analysis.



Functions:

Dependencies:

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import re
from numpy.linalg import svd
from event import Event


def load_events(folder_path):
    """
    load all events in folder_path into Event objects
    :param folder_path: path to event folder
    :return: list of events
    """
    event_objects = []
    files = os.listdir(folder_path)

    for f in files:
        file_path = os.path.join(folder_path, f)
        image = np.load(file_path)
        event = Event(f, image)
        event_objects.append(event)

    return event_objects
