"""
Module that contains standalone functions for image preprocessing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from convert_sim_ims import convert_im, get_dark_sample
import matplotlib.pyplot as plt


def gaussian_smoothing(image, smoothing_sigma=3.5):
    """
    Apply Gaussian smoothing to an event image.

    Parameters
    ----------
    image : np.ndarray
        The image to smooth.
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel, controlling the smoothing level (default is 3.5).

    Returns
    -------
    np.ndarray
        The smoothed image.

    """
    return gaussian_filter(image, sigma=smoothing_sigma)


def smoothing_widget(image, smoothing_sigma):
    """
    Apply Gaussian smoothing to the image and display the result.

    Parameters:
    ----------
    image : np.ndarray
        Input 2D array representing the image.
    smoothing_sigma : float
        Standard deviation for Gaussian kernel.
    """
    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(image, sigma=smoothing_sigma)

    # Calculate total intensities
    original_intensity = np.sum(image)
    smoothed_intensity = np.sum(smoothed_image)

    # Plot the original and smoothed images
    plt.figure(figsize=(10, 10))

    # Original image
    plt.subplot(2, 1, 1)
    plt.imshow(image, cmap="viridis", origin="lower")
    plt.title("Original Image")
    plt.colorbar(label="Intensity")
    plt.grid(False)
    plt.xlabel(f"Total Intensity: {original_intensity:.2f}")

    # Smoothed image
    plt.subplot(2, 1, 2)
    plt.imshow(smoothed_image, cmap="viridis", origin="lower")
    plt.title(f"Smoothed Image (σ={smoothing_sigma})")
    plt.colorbar(label="Intensity")
    plt.grid(False)
    plt.xlabel(f"Total Intensity: {smoothed_intensity:.2f}")

    plt.tight_layout()
    plt.show()


def noise_adder(image, m_dark, example_dark_list, noise_index=None):
    """
    Add noise to an event image based on a master dark image and random sample from example darks.

    Parameters
    ----------
    image : np.ndarray
        The image to which noise will be added.
    m_dark : np.ndarray
        2D array (image) containing the master dark.
    example_dark_list : list of np.ndarray
        List of example dark images from which a random sample is selected for noise addition.
    noise_index : int, optional
        The index of the example dark to use for noise addition (if you want to specify it).


    Returns
    -------
    np.ndarray
        The image with noise added.

    """
    if noise_index is None:
        noise_index = np.random.randint(0, len(example_dark_list) - 1)

    im_dims = [len(image[0]), len(image)]

    noised_image = convert_im(
        image,
        get_dark_sample(
            m_dark,
            im_dims,
            example_dark_list[noise_index],
        ),
    )
    return noised_image


def image_threshold_widget(image, threshold_percentile):
    """
    Apply a threshold to the image based on the given percentile
    and display the original and thresholded images.

    Parameters:
    ----------
    image : np.ndarray
        The input image array to process.
    threshold_percentile : float
        The percentile used to calculate the threshold value.
    """
    threshold = np.percentile(image, threshold_percentile)
    thresholded_image = np.where(image > threshold, image, 0)

    original_intensity = np.sum(image)
    thresholded_intensity = np.sum(thresholded_image)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="viridis", origin="lower")
    plt.title("Original Image")
    plt.colorbar(label="Intensity")
    plt.grid(False)
    plt.xlabel(f"Total Intensity: {original_intensity:.2f}")

    plt.subplot(1, 2, 2)
    plt.imshow(thresholded_image, cmap="gray", origin="lower")
    plt.title(f"Thresholded Image\n(Threshold: {threshold:.2f})")
    plt.colorbar(label="Binary Mask")
    plt.grid(False)
    plt.xlabel(f"Thresholded Intensity: {thresholded_intensity:.2f}")

    plt.tight_layout()
    plt.show()


def uncropped_check(image: np.ndarray, search_fraction: float, method='max_comparison') -> bool:
    """
    Check if an image is uncropped along the x or y axes based on a search fraction.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    search_fraction (float): Fraction of the image size to use as the threshold for cropping.
    method (str): Method to use for uncropped check. Options are 'max_comparison' and 'area_comparison'.

    Returns:
    bool: True if the image is uncropped along either x or y axis, otherwise False.
    """
    # Get the dimensions of the image
    max_y, max_x = image.shape

    # Determine a threshold intensity to avoid floating point errors
    non_zero_values = image[image > 0]
    if non_zero_values.size == 0:
        return False  # No non-zero pixels, so cannot determine uncropped status

    threshold = np.percentile(non_zero_values, 1)  # Smallest 1% of non-zero intensities

    # Find the largest x and y coordinates with intensity greater than the threshold
    nonzero_indices = np.argwhere(image > threshold)
    if nonzero_indices.size == 0:
        return False  # No valid indices after applying threshold

    max_nonzero_y = nonzero_indices[:, 0].max() + 1  # Add 1 to account for indexing
    max_nonzero_x = nonzero_indices[:, 1].max() + 1

    # Compare max_nonzero_x and max_nonzero_y with thresholds
    uncropped_x = max_nonzero_x < search_fraction * max_x
    uncropped_y = max_nonzero_y < search_fraction * max_y

    return uncropped_x or uncropped_y


def dim_check(image: np.ndarray, min_dim: int) -> bool:
    """
    Check if an image has dimensions larger than a minimum value.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    min_dim (int): Minimum dimension in either x or y direction.

    Returns:
    bool: True if the either x or y dimension is less than the minimum value, otherwise False.
    """
    y_dim, x_dim = image.shape
    return y_dim < min_dim or x_dim < min_dim