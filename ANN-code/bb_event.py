"""
This module contains the `Event` class, which endows a nuclear recoil image with attributes.
"""

import json
import re
import matplotlib.pyplot as plt
import os
import numpy as np


with open("matplotlibrc.json", "r") as file:
    custom_params = json.load(file)

plt.rcParams.update(custom_params)


class Event:
    """
    Holds event attributes
    """

    def __init__(self, name, image):
        """
        Initialize an Event instance.

        Parameters:
        ----------
        name : str
            The filename of the event
        image : np.ndarray
            2D array representing the image data of the event. Could be raw or processed.
        """

        self.name = name
        self.image = image
        self.species = self.get_species_from_name()

    def get_energy_from_name(self):
        """
        Extract the energy in keV from the filename.

        Returns:
        -------
        float
            The energy in keV.
        """
        match = re.search(r"(\d+\.?\d*)keV", self.name)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Could not extract energy.")

    def get_species_from_name(self):
        """
        Extract the species (C for carbon or F for fluorine) from the filename.

        Returns:
        -------
        str
            The species (either 'C' or 'F').
        """

        if "C" in self.name:
            return "C"

        if "F" in self.name:
            return "F"

        raise ValueError("Could not extract species.")

    def plot_image(self):
        """
        Display the event image.
        """

        image = self.image
        energy = self.get_energy_from_name()
        plot_name = str(energy) + "keV" + " " + self.species

        fig, ax = plt.subplots()

        ax.grid(False)
        ax.imshow(image)

        plt.title(plot_name)
        plt.show()


def load_events_bb(file_path):
    """
    load barebones events from folder
    """

    events = []

    # Iterate over all .npy files in the directory
    for filename in os.listdir(file_path):
        if filename.endswith(".npy"):
            # Construct full file path
            full_path = os.path.join(file_path, filename)

            # Load image data from .npy file
            image = np.load(full_path)

            # Instantiate Event object
            event = Event(filename, image)

            # Append to events list
            events.append(event)

    return events
