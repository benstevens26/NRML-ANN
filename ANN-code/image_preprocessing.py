"""
Module that contains standalone functions for image preprocessing.
"""

from scipy.ndimage import gaussian_filter

def gaussian_smoothing(image, smoothing_sigma=5):
    """
    Apply Gaussian smoothing to an event image.

    Parameters
    ----------
    image : np.ndarray
        The image to smooth.
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel, controlling the smoothing level (default is 5).

    Returns
    -------
    np.ndarray
        The smoothed image.
    
    """
    return gaussian_filter(image, sigma=smoothing_sigma)
