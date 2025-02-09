"""
Module that contains standalone functions for imaging analysis.
"""

import matplotlib.pyplot as plt
import numpy as np


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
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with intensity as z-height
    ax.plot_surface(x, y, image, cmap="viridis", edgecolor="none")

    # Label the axes
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity (Z-height)")
    ax.set_title("3D Intensity Plot")

    # Show the plot
    plt.show()
