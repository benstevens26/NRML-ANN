"""
Module that contains standalone functions for feature extraction from Event images.
"""

import numpy as np


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


def extract_axis(image, batch=False):
    """
    Efficiently extracts the principal axis of a recoil image using eigen decomposition.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        

    Returns:
        principal_axis (numpy.ndarray): Unit vector indicating the direction of the principal axis.
        centroid (tuple): The coordinates of the image centroid.
    """
    from scipy.ndimage import center_of_mass

    # find centroid
    centroid = center_of_mass(image)
    y, x = np.indices(image.shape)
    centered_x = x - centroid[1]
    centered_y = y - centroid[0]

    # mask out zero intensity pixels (perhaps consider a threshold)
    mask = image > 0
    weights = image[mask]
    centered_coords = np.stack([centered_x[mask], centered_y[mask]], axis=1)

    weighted_coords = centered_coords * weights[:, None]
    cov_matrix = np.cov(weighted_coords.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    return principal_axis, centroid


