"""
This module provides the `Event` class, which represents a nuclear recoil event with attributes
such as the event's name, image, energy, species, and track length.

Classes:
    - Event: Represents a nuclear recoil event with attributes such as name, image, energy, species,
      and length. Provides methods to process the event image, extract features such as species and energy,
      and compute the principal axis using image analysis.

Methods in Event:
    - __init__(self, name, image, smoothing=5): Initializes an Event object with the event name
      and raw image data. It also applies Gaussian smoothing to the image and extracts the
      energy, species, and track length from the event name.

    - get_energy_from_name(self): Extracts the energy in keV from the event's filename.

    - get_species_from_name(self): Extracts the species (C for carbon, F for fluorine)
      from the event's filename.

    - get_length_from_name(self): Extracts the track length in cm from the event's filename.

    - get_attributes(self): Returns the key attributes of the event, including name, species, energy,
      length, and processed image.

    - get_smoothed_image(self, smoothing_sigma): Applies Gaussian smoothing to the raw image data
      to reduce noise.

    - extract_principal_axis(self, plot=False): Calculates the principal axis of the event
      using Singular Value Decomposition (SVD) on the image data. Optionally plots the image
      with the principal axis overlaid.

Additional setup:
    - The module loads custom Matplotlib plotting parameters from a JSON file, which is used
      to globally update Matplotlib's settings.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from numpy.linalg import svd
from tqdm import tqdm

with open("matplotlibrc.json", "r") as file:
    custom_params = json.load(file)

plt.rcParams.update(custom_params)


class Event:
    """ """

    def __init__(self, name, image):
        self.name = name
        self.image = image

        self.species = self.get_species_from_name()
        self.energy = self.get_energy_from_name()
        self.depth = self.get_depth_from_name()

        self.plot_name = str(self.energy) + "keV" + " " + self.species

        self.principal_axis = None
        self.bisectors = None
        self.mean_x = None
        self.mean_y = None

    def get_energy_from_name(self):
        """
        Extract the energy in keV from the filename.
        :return: The energy in keV as a float.
        """
        match = re.search(r"(\d+\.?\d*)keV", self.name)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Could not extract energy.")

    def get_species_from_name(self):
        """
        Extract the species (C for carbon or F for fluorine) from the filename.
        :return: The species as a string.
        """
        match = re.search(r"_([C|F])_", self.name)
        if match:
            return match.group(1)
        else:
            raise ValueError("Could not extract species.")

    def get_depth_from_name(self):
        """
        Extract the length in cm from the filename.
        :return: The length in cm as a float.
        """
        match = re.search(r"(\d+\.?\d*)cm", self.name)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Could not extract length.")

    def get_attributes(self):
        return self.name, self.energy, self.species, self.length

    def get_principal_axis(self):
        """
        Extract principal axis from image

        :return: list [principle axis, mean_x, mean_y]
        """

        image = self.image
        height, width = image.shape

        # Create a grid of coordinates for each pixel
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten the arrays to get a list of (x, y) points
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        intensities_flat = image.flatten()

        # Filter out zero intensities
        non_zero = intensities_flat > 0
        x_flat = x_flat[non_zero]
        y_flat = y_flat[non_zero]

        # Create a matrix where rows represent pixel positions, and columns are the coordinates weighted by intensity
        data_matrix = np.vstack((x_flat, y_flat)).T

        # Perform SVD on the data matrix
        U, S, Vt = svd(data_matrix - np.mean(data_matrix, axis=0), full_matrices=False)

        # The principal axis is given by the first row of Vt (right singular vectors)
        principal_axis = Vt[0]

        if principal_axis[0] < 0:
            principal_axis[0] = -principal_axis[0]

        mean_x = np.mean(x_flat)
        mean_y = np.mean(y_flat)

        return principal_axis, mean_x, mean_y

    def get_track_length(
        self, num_segments, segment_distances=None, segment_intensities=None
    ):
        """
        Calculate the track length based on segment distances and segment intensities.

        Parameters:
        - segment_distances: List of cumulative distances along the principal axis.
        - segment_intensities: List of total intensities for each segment.

        Returns:
        - track_length: The calculated length of the track.
        """

        if segment_distances is None:
            segment_distances, segment_intensities = self.get_intensity_profile(
                num_segments
            )

        non_zero_indices = [
            i for i, intensity in enumerate(segment_intensities) if intensity > 0
        ]

        if not non_zero_indices:
            # If there are no non-zero segments, return zero length
            return 0.0

        # First and last non-zero intensity segment indices
        first_non_zero_index = non_zero_indices[0]
        last_non_zero_index = non_zero_indices[-1]

        # Calculate the midpoints of the first and last non-zero segments
        midpoint_first = (
            segment_distances[first_non_zero_index]
            + segment_distances[first_non_zero_index + 1]
        ) / 2

        if last_non_zero_index < len(segment_distances) - 1:
            midpoint_last = (
                segment_distances[last_non_zero_index]
                + segment_distances[last_non_zero_index + 1]
            ) / 2
        else:
            # If last_non_zero_index is the last segment, use only that segment's distance
            midpoint_last = segment_distances[last_non_zero_index]
        # Calculate the track length as the distance between the two midpoints
        track_length = abs(midpoint_last - midpoint_first)

        return track_length

    def get_track_energy(self):
        raise NotImplementedError("This function is not yet implemented")

    def get_max_den(self):

        max_den = 1 / max(self.image)
        return max_den

    def get_recoil_angle(self, principal_axis=None):
        """
        Calculate the angle between the principal axis and the +x direction.
        """

        x_direction = np.array([1, 0])

        # Normalize the principal axis vector
        principal_axis = np.array(principal_axis)
        principal_axis_norm = principal_axis / np.linalg.norm(principal_axis)

        # Calculate the angle using the dot product formula
        dot_product = np.dot(principal_axis_norm, x_direction)
        angle_rad = np.arccos(dot_product)  # Angle in radians

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_bisectors(self, num_segments):
        """ """

        image = self.image

        if self.principal_axis is None:
            self.principal_axis, self.mean_x, self.mean_y = self.get_principal_axis()

        principal_axis, mean_x, mean_y = self.principal_axis, self.mean_x, self.mean_y

        height, width = image.shape

        # Calculate the length of the principal axis extended over the whole image
        line_length = np.sqrt(width**2 + height**2)

        # Extend the principal axis over the full image dimensions
        x_start = mean_x - principal_axis[0] * line_length / 2
        y_start = mean_y - principal_axis[1] * line_length / 2
        x_end = mean_x + principal_axis[0] * line_length / 2
        y_end = mean_y + principal_axis[1] * line_length / 2

        # Divide the principal axis into equal segments
        segment_points_x = np.linspace(x_start, x_end, num_segments + 1)
        segment_points_y = np.linspace(y_start, y_end, num_segments + 1)

        # List to store bisectors' start and end points
        bisectors = []

        # Calculate the perpendicular bisectors
        for i in range(num_segments + 1):
            # Get the current segment boundary point
            x_point = segment_points_x[i]
            y_point = segment_points_y[i]

            # Perpendicular vector direction (rotate 90 degrees)
            perp_vector = np.array([-principal_axis[1], principal_axis[0]])

            # Extend the perpendicular bisector for calculation
            perp_length = line_length / 2  # Length of perpendicular bisector
            x_bisector_start = x_point - perp_vector[0] * perp_length
            y_bisector_start = y_point - perp_vector[1] * perp_length
            x_bisector_end = x_point + perp_vector[0] * perp_length
            y_bisector_end = y_point + perp_vector[1] * perp_length

            # Append the bisector start and end points as a tuple
            bisectors.append(
                ((x_bisector_start, y_bisector_start), (x_bisector_end, y_bisector_end))
            )

        return bisectors

    def get_intensity_profile(self, num_segments):

        image = self.image
        self.bisectors = self.get_bisectors(num_segments)
        bisectors = self.bisectors

        segment_intensities = np.zeros(num_segments)

        # Identify non-zero pixels
        non_zero_y, non_zero_x = np.nonzero(
            image
        )  # Get y and x indices of non-zero pixels
        non_zero_pixels = zip(
            non_zero_x, non_zero_y
        )  # Create a list of (x, y) tuples for non-zero pixels

        # Function to calculate the cross product of two vectors
        def cross_product_2d(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        # Loop through each non-zero pixel
        for x, y in non_zero_pixels:
            # Get the intensity value of the current pixel
            intensity = image[y, x]

            # Convert pixel position to array
            pixel_pos = np.array([x, y])

            # Check which segment this pixel belongs to by using cross products
            for i in range(num_segments):
                # Get current bisector and the next bisector (defining the segment boundaries)
                (x_bisector_start, y_bisector_start), (
                    x_bisector_end,
                    y_bisector_end,
                ) = bisectors[i]
                (x_next_bisector_start, y_next_bisector_start), (
                    x_next_bisector_end,
                    y_next_bisector_end,
                ) = bisectors[i + 1]

                # Define the vectors from the pixel to the start of the bisectors
                vector_to_bisector = pixel_pos - np.array(
                    [x_bisector_start, y_bisector_start]
                )
                vector_to_next_bisector = pixel_pos - np.array(
                    [x_next_bisector_start, y_next_bisector_start]
                )

                # Define the direction vectors of the bisectors
                bisector_vector = np.array(
                    [
                        x_bisector_end - x_bisector_start,
                        y_bisector_end - y_bisector_start,
                    ]
                )
                next_bisector_vector = np.array(
                    [
                        x_next_bisector_end - x_next_bisector_start,
                        y_next_bisector_end - y_next_bisector_start,
                    ]
                )

                # Calculate the cross products to check which side of the bisector the pixel is on
                cross_current = cross_product_2d(bisector_vector, vector_to_bisector)
                cross_next = cross_product_2d(
                    next_bisector_vector, vector_to_next_bisector
                )

                # Check if the pixel lies between the two bisectors
                if cross_current <= 0 <= cross_next:
                    # Add the pixel intensity to the corresponding segment
                    segment_intensities[i] += intensity
                    break  # Once classified, move to the next pixel

        # Calculate actual distances along the principal axis
        segment_distances = []
        total_distance = 0

        for i in range(num_segments):
            # Get the current segment's start and next bisector
            (x_bisector_start, y_bisector_start), (x_bisector_end, y_bisector_end) = (
                bisectors[i]
            )
            (x_next_bisector_start, y_next_bisector_start), (
                x_next_bisector_end,
                y_next_bisector_end,
            ) = bisectors[i + 1]

            # Calculate the distance between current bisector and next bisector
            distance = np.sqrt(
                (x_next_bisector_start - x_bisector_start) ** 2
                + (y_next_bisector_start - y_bisector_start) ** 2
            )
            total_distance += distance
            segment_distances.append(total_distance)

        return segment_distances, segment_intensities

    def plot_image(self):

        image = self.image

        fig, ax = plt.subplots()

        ax.grid(False)
        ax.imshow(image)

        plt.title(self.plot_name + " image")
        plt.show()

    def plot_image_with_principal_axis(self):
        """ """

        image = self.image

        if self.principal_axis is None:
            self.principal_axis, self.mean_x, self.mean_y = self.get_principal_axis()

        height, width = image.shape

        # Calculate the length of the principal axis extended over the whole image
        line_length = np.sqrt(
            width**2 + height**2
        )  # Diagonal length to ensure it covers the whole image

        # Extend the principal axis over the full image dimensions
        x_start = self.mean_x - self.principal_axis[0] * line_length / 2
        y_start = self.mean_y - self.principal_axis[1] * line_length / 2
        x_end = self.mean_x + self.principal_axis[0] * line_length / 2
        y_end = self.mean_y + self.principal_axis[1] * line_length / 2

        # Plot the image

        fig, ax = plt.subplots()

        ax.grid(False)
        ax.imshow(image)
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="red",
            label="Principal Axis",
            linewidth=2,
        )
        plt.title(self.plot_name + " with principal axis")
        plt.legend()
        plt.show()

    def plot_image_with_bisectors(self, num_segments):
        """ """

        image = self.image

        if self.bisectors is None:
            self.bisectors = self.get_bisectors(num_segments)
        bisectors = self.bisectors

        principal_axis, mean_x, mean_y = self.principal_axis, self.mean_x, self.mean_y

        # Calculate the length of the principal axis extended over the whole image
        height, width = image.shape
        line_length = np.sqrt(width**2 + height**2)

        # Extend the principal axis over the full image dimensions
        x_start = mean_x - principal_axis[0] * line_length / 2
        y_start = mean_y - principal_axis[1] * line_length / 2
        x_end = mean_x + principal_axis[0] * line_length / 2
        y_end = mean_y + principal_axis[1] * line_length / 2

        fig, ax = plt.subplots()
        plt.imshow(image, origin="lower")
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="red",
            label="Principal Axis",
            linewidth=2,
        )

        for bisector in bisectors:
            (x_bisector_start, y_bisector_start), (
                x_bisector_end,
                y_bisector_end,
            ) = bisector
            ax.plot(
                [x_bisector_start, x_bisector_end],
                [y_bisector_start, y_bisector_end],
                "blue",
                linestyle="-",
                label="Bisector" if bisector == bisectors[0] else "",
            )

        plt.title(self.plot_name + " with principal axis and bisectors")
        plt.legend()
        plt.show()

    def plot_intensity_profile(
        self, num_segments, segment_distances=None, segment_intensities=None
    ):
        """ """

        if segment_distances is None:
            segment_distances, segment_intensities = self.get_intensity_profile(
                num_segments
            )
        # Plot the intensity profile as a function of the actual distance along the principal axis
        plt.bar(
            segment_distances,
            segment_intensities,
            width=segment_distances[1] - segment_distances[0],
        )
        plt.xlabel("Distance Along Principal Axis (pixels)")
        plt.ylabel("Total Intensity")
        plt.title(self.plot_name + " with intensity profile")
        plt.show()


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
        image = get_smoothed_image(np.load(file_path), smoothing_sigma=5)
        event = Event(f, image)
        event_objects.append(event)

    return event_objects


def get_smoothed_image(image, smoothing_sigma):
    return nd.gaussian_filter(image, sigma=smoothing_sigma)
