"""
Module that contains standalone functions for imaging analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display
from feature_extraction import extract_axis, extract_intensity_profile, extract_length


def plot_axis(image, principal_axis, centroid):
    """
    Plots the image with the principal axis overlayed.

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

    # Mark the centroid
    plt.scatter(mean_x, mean_y, color="blue", label="Centroid")

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Image with Principal Axis")
    plt.legend()

    plt.show()


def plot_intensity_contour(image, grid_x, grid_y, grid_z):
    """
    Plots the original image with the grid interpolation overlayed.

    Parameters:
        image (numpy.ndarray): 2D array representing the original image.
        grid_x (numpy.ndarray): X-coordinates of the interpolated grid.
        grid_y (numpy.ndarray): Y-coordinates of the interpolated grid.
        grid_z (numpy.ndarray): Interpolated intensity values from the grid.
    """
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(
        image,
        cmap="viridis",
        origin="lower",
        extent=(0, image.shape[1], 0, image.shape[0]),
    )
    plt.colorbar(label="Intensity")
    plt.title("Original Image")

    # Plot the interpolated spline
    plt.subplot(1, 2, 2)
    plt.imshow(
        grid_z,
        cmap="viridis",
        origin="lower",
        extent=(0, image.shape[1], 0, image.shape[0]),
    )
    plt.colorbar(label="Interpolated Intensity")
    plt.contour(grid_x, grid_y, grid_z, levels=10, colors="red", linewidths=0.5)
    plt.title("Grid Interpolation")

    # Add common labels
    plt.suptitle("Image with Grid Interpolation", fontsize=16)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.tight_layout()

    plt.show()


def plot_spline(image, x_spline, y_spline):
    """
    Plots the original image with the principal axis spline overlayed.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        x_spline (numpy.ndarray): Interpolated x-coordinates of the spline.
        y_spline (numpy.ndarray): Interpolated y-coordinates of the spline.
    """
    plt.figure(figsize=(8, 8))

    # Plot the original image
    plt.imshow(
        image,
        cmap="viridis",
        origin="lower",
        extent=(0, image.shape[1], 0, image.shape[0]),
    )
    plt.colorbar(label="Intensity")

    # Overlay the spline principal axis
    plt.plot(
        x_spline,
        y_spline,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Principal Axis Spline",
    )

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Image with Principal Axis Spline")
    plt.legend()

    plt.show()


def plot_bounding_box(image: np.ndarray, bounding_box: tuple):
    """
    Plot the image with the bounding box overlayed.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    bounding_box (tuple): The bounding box coordinates (min_y, min_x, max_y, max_x).
    """
    if bounding_box is None:
        print("No bounding box to plot.")
        return

    min_y, min_x, max_y, max_x = bounding_box

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="viridis")
    plt.gca().add_patch(
        plt.Rectangle(
            (min_x, min_y),
            max_x - min_x + 1,
            max_y - min_y + 1,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
    )
    plt.title("Image with Bounding Box")
    plt.show()


def plot_binary_image(image: np.ndarray):
    """
    Plot an image where all non-zero intensities are set to 1.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    """
    binary_image = (image > 0).astype(int)
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_image, cmap="gray")
    plt.title("Non-zero Intensities (Binary Representation)")
    plt.show()


def plot_3d(image: np.ndarray) -> None:
    """
    Plots a 3D graph where the z-height represents the intensity of the image.

    Parameters:
        image (numpy.ndarray): 2D array representing the image, where each element is the intensity.

    Returns:
        None
    """
    # Get the dimensions of the image
    height, width = image.shape

    # Create a grid of x, y coordinates
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with intensity as z-height
    ax.plot_surface(x, y, image, cmap='viridis', edgecolor='none')

    # Label the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Intensity (Z-height)')
    ax.set_title('3D Intensity Plot')

    # Show the plot
    plt.show()


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
