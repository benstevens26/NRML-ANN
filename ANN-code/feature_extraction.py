# feature extraction for input into the ANN

import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import os

folder_path = os.path.join("..", "..", "Data/C/300-320keV")
# folder_path = os.path.join( tom path )

files = os.listdir(folder_path)
carbon_events = [np.load(folder_path + "/" + f) for f in files]


def extract_axis(image, plot=False, return_extras=False):
    """
    Extract principle axis from image

    :param image:
    :param extras: if True, return extras
    :param plot: if True, plot the image overlayed with principle axis
    :return: principle axis
    """

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
        plt.show()

    if return_extras:
        # space to return extras if wanted

        return principal_axis

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


for image in carbon_events:
    extract_axis(image, plot=True)

