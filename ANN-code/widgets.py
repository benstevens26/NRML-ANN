"""
Widgets
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider
from IPython.display import display
from scipy.ndimage import gaussian_filter

import os
print(os.getcwd())

from feature_extraction import extract_axis, extract_intensity_profile, extract_length



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



def intensity_profile_widget(image: np.ndarray):
    """
    Creates an interactive widget to explore how the extracted intensity profile depends on Gaussian smoothing.

    Parameters:
        image (numpy.ndarray): 2D array representing the unsmoothed image.

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

    def update_plot(smoothing_sigma):
        smoothed_image = gaussian_smoothing(image, smoothing_sigma)

        # Plot the smoothed image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(smoothed_image, cmap='viridis', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f"Smoothed Image (Smoothing σ={smoothing_sigma})")
        plt.axis('off')

        # Extract and plot the intensity profile
        distances, intensities = extract_intensity_profile(smoothed_image, plot=False)
        plt.subplot(1, 2, 2)
        plt.plot(distances, intensities, label="Intensity")
        plt.axvline(0, color='red', linestyle='--', label='Centroid')
        plt.title("Intensity Along Principal Axis")
        plt.xlabel("Distance Along Principal Axis")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    interactive_plot = widgets.interactive(update_plot, smoothing_sigma=smoothing_slider)
    display(interactive_plot)




def recoil_length_widget(image: np.ndarray):
    """
    Creates an interactive widget to visualize the intensity profile and recoil track with adjustable threshold.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.

    Returns:
        None
    """


    def extract_length_widget(image: np.ndarray, energy_percentile: float) -> tuple:

        # Extract intensity profile
        distances, intensities = extract_intensity_profile(image, plot=False)

        # Calculate the threshold intensity based on the percentile
        threshold = np.percentile(intensities, energy_percentile)

        # Find indices where intensity exceeds the threshold
        above_threshold = np.where(np.array(intensities) >= threshold)[0]

        if len(above_threshold) == 0:
            return 0.0, None, None  # No recoil above the threshold

        # Calculate the length of the recoil
        min_t = distances[above_threshold[0]]
        max_t = distances[above_threshold[-1]]
        recoil_length = max_t - min_t

        # Get the start and end points in physical coordinates
        principal_axis, centroid = extract_axis(image)
        start_point = (
            centroid[0] + min_t * principal_axis[0],
            centroid[1] + min_t * principal_axis[1],
        )
        end_point = (
            centroid[0] + max_t * principal_axis[0],
            centroid[1] + max_t * principal_axis[1],
        )

        return recoil_length, start_point, end_point

    threshold_slider = widgets.FloatSlider(
        value=50.0,
        min=0.0,
        max=100.0,
        step=1.0,
        description='Threshold (%)',
        continuous_update=False
    )

    def update_visualization(energy_percentile):
        recoil_length, start_point, end_point = extract_length_widget(image, energy_percentile)
        distances, intensities = extract_intensity_profile(image, plot=False)

        # Calculate the threshold value
        threshold = np.percentile(intensities, energy_percentile)

        # Create the figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot intensity profile
        axes[0].plot(distances, intensities, label="Intensity")
        axes[0].axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
        axes[0].set_title(f"Intensity Profile\nRecoil Length = {recoil_length:.2f} pixels")
        axes[0].set_xlabel("Distance Along Principal Axis")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()
        axes[0].grid()

        # Plot recoil track with principal axis and start/end points
        axes[1].imshow(image, cmap='gray', origin='lower')
        principal_axis, centroid = extract_axis(image)
        axes[1].quiver(
            centroid[0], centroid[1],
            principal_axis[0], principal_axis[1],
            angles='xy', scale_units='xy', scale=1, color='red', label='Principal Axis'
        )
        if start_point and end_point:
            axes[1].plot(
                [start_point[0], end_point[0]], [start_point[1], end_point[1]],
                color='blue', linestyle='--', label='Recoil Track'
            )
            axes[1].scatter(*start_point, color='green', label='Start Point')
            axes[1].scatter(*end_point, color='orange', label='End Point')
        axes[1].set_title("Recoil Track")
        axes[1].legend()
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    interactive_plot = widgets.interactive(update_visualization, energy_percentile=threshold_slider)
    display(interactive_plot)
