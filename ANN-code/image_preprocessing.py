"""
Module that contains standalone functions for image preprocessing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from convert_sim_ims import convert_im, get_dark_sample
from feature_extraction import extract_bounding_box
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider
from image_analysis import plot_axis
from feature_extraction import extract_axis
from image_analysis import plot_3d


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


def smoothing_widget_3d(image: np.ndarray):
    """
    Creates an interactive widget to plot the 3D image with adjustable Gaussian smoothing.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.

    Returns:
        None
    """
    smoothing_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description='Smoothing σ:',
        continuous_update=False
    )

    interactive_plot = widgets.interactive(
        lambda sigma: plot_3d(gaussian_smoothing(image, sigma)),
        sigma=smoothing_slider
    )

    display(interactive_plot)



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


def uncropped_check(
    image: np.ndarray, search_fraction: float, method: str = "max_comparison"
) -> bool:
    """
    Check if an image is uncropped using different methods based on a search fraction.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    search_fraction (float): Depends on the method used. For 'max_comparison', it is the fraction of the image dimensions.
    method (str): The method to use for the check. Options are 'max_comparison' or 'area_comparison'.

    Returns:
    bool: True if the image is uncropped based on the selected method, otherwise False.
    """
    # Get the dimensions of the image
    max_y, max_x = image.shape

    if method == "max_comparison":
        # Determine a threshold intensity to avoid floating point errors
        non_zero_values = image[image > 0]
        if non_zero_values.size == 0:
            return False  # No non-zero pixels, so cannot determine uncropped status

        threshold = np.percentile(
            non_zero_values, 1
        )  # Smallest 1% of non-zero intensities

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

    elif method == "area_comparison":
        # extract the bounding box
        bounding_box = extract_bounding_box(image)

        # calculate the area of the bounding box
        box_area = (bounding_box[2] - bounding_box[0]) * (
            bounding_box[3] - bounding_box[1]
        )

        # calculate the area of the image
        image_area = max_y * max_x

        # return True if the box area is less than the search fraction of the image area
        return box_area / image_area < search_fraction

    else:
        raise ValueError(
            "Invalid method. Choose 'max_comparison' or 'area_comparison'."
        )


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


def zero_edges(image: np.ndarray, edge_width: int = 1) -> np.ndarray:
    """
    Zero the intensities in the specified number of pixels from each edge of the image.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    edge_width (int): The width of the edges to zero out.

    Returns:
    np.ndarray: The image with edges zeroed.
    """
    # Copy the image to avoid modifying the original
    image_zeroed = image.copy()

    # Zero out the edges
    image_zeroed[:edge_width, :] = 0  # Top edge
    image_zeroed[-edge_width:, :] = 0  # Bottom edge
    image_zeroed[:, :edge_width] = 0  # Left edge
    image_zeroed[:, -edge_width:] = 0  # Right edge

    return image_zeroed


def plot_axis_widget(image: np.ndarray, principal_axis: np.ndarray, centroid: tuple):
    """
    Plots the image with the principal axis overlayed (for widget use).

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        principal_axis (numpy.ndarray): Array containing the principal axis vector [x, y].
        centroid (tuple): Coordinates of the centroid (mean_x, mean_y).
    """
    height, width = image.shape
    mean_x, mean_y = centroid

    # Principal axis vector components
    dx, dy = principal_axis

    # Normalize the principal axis to scale across the image dimensions
    if abs(dx) > abs(dy):
        scale = width / (2 * abs(dx))
    else:
        scale = height / (2 * abs(dy))

    x_start = mean_x - dx * scale
    x_end = mean_x + dx * scale
    y_start = mean_y - dy * scale
    y_end = mean_y + dy * scale

    # Clip the line to image boundaries
    x_start, x_end = np.clip([x_start, x_end], 0, width)
    y_start, y_end = np.clip([y_start, y_end], 0, height)

    # Plot the image
    plt.imshow(image, cmap="viridis", origin="lower", extent=(0, width, 0, height))

    # Overlay the principal axis
    plt.plot(
        [x_start, x_end],
        [y_start, y_end],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Principal Axis",
    )


def combined_widget(image: np.ndarray, axis=False):
    """
    Creates an interactive widget to apply both Gaussian smoothing and thresholding
    to an image with sliders for sigma and threshold percentile.

    Parameters:
    ----------
    image : np.ndarray
        Input 2D array representing the image.
    """
    if axis:

        def process_image(smoothing_sigma: float, threshold_percentile: float):
            """
            Process the image with Gaussian smoothing and thresholding.

            Parameters:
            ----------
            smoothing_sigma : float
                Standard deviation for Gaussian kernel.
            threshold_percentile : float
                Percentile for thresholding the image.
            """
            # Apply Gaussian smoothing
            smoothed_image = gaussian_filter(image, sigma=smoothing_sigma)

            # Apply thresholding
            threshold_value = np.percentile(smoothed_image, threshold_percentile)
            thresholded_image = np.where(smoothed_image >= threshold_value, smoothed_image, 0)

            # Calculate principal axis and centroid for smoothed image
            smoothed_axis, smoothed_centroid = extract_axis(smoothed_image)

            # Calculate principal axis and centroid for thresholded image
            thresholded_axis, thresholded_centroid = extract_axis(thresholded_image)

            # Plot the original, smoothed, and thresholded images with principal axes
            plt.figure(figsize=(15, 5))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap="viridis", origin="lower")
            plt.title("Original Image")
            plt.grid(False)

            # Smoothed image with principal axis
            plt.subplot(1, 3, 2)
            plot_axis_widget(smoothed_image, smoothed_axis, smoothed_centroid)
            plt.title(f"Smoothed Image (σ={smoothing_sigma})")

            # Thresholded image with principal axis
            plt.subplot(1, 3, 3)
            plot_axis_widget(thresholded_image, thresholded_axis, thresholded_centroid)
            plt.title(f"Thresholded Image\n(>{threshold_percentile}th Percentile)")

            plt.tight_layout()
            plt.show()

        # Create interactive sliders for smoothing_sigma and threshold_percentile
        interact(
            process_image,
            smoothing_sigma=FloatSlider(value=1.0, min=0, max=10, step=0.1, description="Smoothing σ"),
            threshold_percentile=FloatSlider(value=95, min=0, max=100, step=1, description="Percentile"),
        )
    else:
        def process_image(smoothing_sigma: float, threshold_percentile: float):
            """
            Process the image with Gaussian smoothing and thresholding.

            Parameters:
            ----------
            smoothing_sigma : float
                Standard deviation for Gaussian kernel.
            threshold_percentile : float
                Percentile for thresholding the image.
            """
            # Apply Gaussian smoothing
            smoothed_image = gaussian_filter(image, sigma=smoothing_sigma)

            # Apply thresholding
            threshold_value = np.percentile(smoothed_image, threshold_percentile)
            thresholded_image = np.where(smoothed_image >= threshold_value, smoothed_image, 0)

            # Plot the original, smoothed, and thresholded images
            plt.figure(figsize=(15, 5))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap="viridis", origin="lower")
            plt.title("Original Image")
            plt.colorbar(label="Intensity")
            plt.grid(False)

            # Smoothed image
            plt.subplot(1, 3, 2)
            plt.imshow(smoothed_image, cmap="viridis", origin="lower")
            plt.title(f"Smoothed Image (σ={smoothing_sigma})")
            plt.colorbar(label="Intensity")
            plt.grid(False)

            # Thresholded image
            plt.subplot(1, 3, 3)
            plt.imshow(thresholded_image, cmap="viridis", origin="lower")
            plt.title(f"Thresholded Image\n(>{threshold_percentile}th Percentile)")
            plt.colorbar(label="Intensity")
            plt.grid(False)

            plt.tight_layout()
            plt.show()

        # Create interactive sliders for smoothing_sigma and threshold_percentile
        interact(
            process_image,
            smoothing_sigma=FloatSlider(value=1.0, min=0, max=10, step=0.1, description="Smoothing σ"),
            threshold_percentile=FloatSlider(value=95, min=0, max=100, step=1, description="Percentile"),
        )
