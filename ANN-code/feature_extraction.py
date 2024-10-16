"""
feature_extraction.py

This module provides functionality to handle and process image data for input into an Artificial Neural Network (ANN).
It includes classes and methods to load events, process images, and extract relevant features for further analysis.

Classes:
    - Event: A class to store image data related to an event.

Functions:
    - load_events(folder_path): Loads image data from the specified folder and returns a list of Event objects.
    - extract_axis(image, plot=False, return_extras=False): Extracts the principal axis of the input image, with optional plotting.

Dependencies:
    - os: Used for file handling.
    - matplotlib.pyplot: For plotting the images.
    - numpy: For numerical operations and loading image data.
    - numpy.linalg.svd: For singular value decomposition in image analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import re
from numpy.linalg import svd


class Event:
    """
    Class to store image information
    """
    def __init__(self, name, image):
        self.name = name
        self.image = image


def load_events(folder_path):
    """
    load all events in folder_path into Event objects
    :param folder_path:
    :return:
    """
    event_objects = []
    files = os.listdir(folder_path)

    for f in files:
        file_path = os.path.join(folder_path, f)
        image_data = np.load(file_path)
        event = Event(f, image_data)
        event_objects.append(event)

    return event_objects


def extract_axis(image, plot=False, return_extras=False):
    """
    Extract principle axis from image

    :param event: event object
    :param extras: if True, return extras
    :param plot: if True, plot the image overlayed with principle axis
    :return: principle axis
    """

    image = event.image
    height, width = image.shape

    # Create a grid of coordinates for each pixel
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays to get a list of (x, y) points
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    intensities_flat = image.flatten()

    # filter out zero intensities
    non_zero = intensities_flat > 0
    x_flat = x_flat[non_zero]
    y_flat = y_flat[non_zero]
    intensities_flat = intensities_flat[non_zero]

    # Create a matrix where rows represent pixel positions, and columns are the coordinates weighted by intensity
    data_matrix = np.vstack((x_flat, y_flat)).T

    # Perform SVD on the data matrix
    U, S, Vt = svd(data_matrix - np.mean(data_matrix, axis=0), full_matrices=False)

    # The principal axis is given by the first row of Vt (right singular vectors)
    principal_axis = Vt[0]

    mean_x = np.mean(x_flat)
    mean_y = np.mean(y_flat)

    # extend the line in both directions from the center of mass
    line_length = max(width, height)  # for plot across whole image

    # Starting and ending points for the line
    x_start = mean_x - principal_axis[0] * line_length / 2
    y_start = mean_y - principal_axis[1] * line_length / 2

    x_end = mean_x + principal_axis[0] * line_length / 2
    y_end = mean_y + principal_axis[1] * line_length / 2

    if plot:
        # Plot the image and the principal axis
        plt.imshow(image, cmap="viridis", origin="lower")
        plt.plot([x_start, x_end], [y_start, y_end], color="red", linewidth=2)
        plt.colorbar()
        plt.show()

    if return_extras:
        return principal_axis, mean_x, mean_y

    return principal_axis


def extract_pixels(image, principal_axis, mean_x, mean_y, threshold=2):
    """
    Extract intensity values for pixels within a certain distance from the principal axis.

    Parameters:
    - image: 2D numpy array of pixel intensities.
    - principal_axis: 2D vector representing the principal axis.
    - mean_x, mean_y: Center point of the principal axis.
    - threshold: Distance in pixels within which to consider pixels "close" to the axis.

    Returns:
    - selected_pixels: List of (x, y, intensity) for pixels near the principal axis.
    """
    height, width = image.shape
    selected_pixels = []

    # Principal axis line equation coefficients: Ax + By + C = 0
    A = -principal_axis[1]
    B = principal_axis[0]
    C = -(A * mean_x + B * mean_y)

    # Loop through all pixels in the image
    for x in range(width):
        for y in range(height):
            # Calculate the distance from the pixel (x, y) to the principal axis
            distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)

            # If the pixel is within the threshold distance from the principal axis, select it
            if distance <= threshold:
                intensity = image[
                    y, x
                ]  # y is the row, x is the column (height x width)
                selected_pixels.append((x, y, intensity))

    return selected_pixels


def plot_deposition(pixels):
    """

    :param pixels: pixels along the principle axis [(x, y, intensity)]
    :return:
    """

    # Extract x, y, intensity values
    x_vals = [point[0] for point in pixels]
    y_vals = [point[1] for point in pixels]
    intensity_vals = [point[2] for point in pixels]

    # Compute distances r along the line
    r_vals = [0]  # The first point is at distance r = 0
    initial_x, initial_y = x_vals[0], y_vals[0]

    for i in range(1, len(pixels)):
        delta_x = x_vals[i] - initial_x
        delta_y = y_vals[i] - initial_y
        r = np.sqrt(delta_x**2 + delta_y**2)
        r_vals.append(r)

    # Plot intensity vs distance
    plt.figure(figsize=(8, 6))
    plt.scatter(r_vals, intensity_vals, marker=".", color="red")
    plt.xlabel("Distance along the PA (r / pixels)")
    plt.ylabel("Intensity")
    plt.title("Intensity vs Distance along PA")
    plt.grid(True)
    plt.show()


def extract_MaxDen(image):
    """
    Extract maximumm
    :param image:
    :return:
    """

    return np.max(image)


def extract_length(image, principal_axis):
    raise NotImplementedError("This function is not yet implemented")

