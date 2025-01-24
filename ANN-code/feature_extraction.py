"""
Module that contains standalone functions for feature extraction from Event images.
"""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.linalg import svd


def extract_energy_deposition(image):
    """
    Extract the total energy deposition from image

    Parameters
    ----------
    image : np.ndarray
        The image to extract the energy deposition from.

    Returns
    -------
    float
        The total energy deposition in the image.
    """
    return np.sum(image)


def extract_max_pixel_intensity(image):
    """
    Extract the maximum pixel intensity from image

    Parameters
    ----------
    image : np.ndarray
        The image to extract the maximum pixel intensity from.

    Returns
    -------
    float
        The maximum pixel intensity in the image.
    """
    return np.max(image)


def extract_axis(image, method="eigen"):
    """
    Extracts the principal axis of a recoil image.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        method (str): Method to use for principal axis extraction. Options are 'eigen' (default) and 'svd'.

    Returns:
        principal_axis (numpy.ndarray): Unit vector indicating the direction of the principal axis.
        centroid (tuple): The coordinates of the image centroid.
    """

    if method == "eigen":

        y_coords, x_coords = np.nonzero(image)
        intensities = image[y_coords, x_coords]

        total_intensity = intensities.sum()
        mean_x = (x_coords * intensities).sum() / total_intensity
        mean_y = (y_coords * intensities).sum() / total_intensity

        centered_x = x_coords - mean_x
        centered_y = y_coords - mean_y

        weighted_coords = np.stack(
            [centered_x * intensities, centered_y * intensities], axis=0
        )
        cov_matrix = np.cov(weighted_coords)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        if principal_axis[0] < 0:
            principal_axis = -principal_axis

        return principal_axis, (mean_x, mean_y)

    elif method == "svd":

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

        return principal_axis, (mean_x, mean_y)
