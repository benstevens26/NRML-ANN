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
