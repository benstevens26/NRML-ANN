"""
Module that contains standalone functions for imaging analysis.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_axis(image, axis, centroid):
    """
    Plots an image with the principal axis overlaid.

    Parameters:
    ----------
    image : np.ndarray
        2D array representing the image.
    principal_axis : np.ndarray
        Unit vector indicating the direction of the principal axis (2 elements).
    centroid : tuple
        Coordinates of the centroid (y, x).
    """
    # Unpack centroid and principal axis
    c_y, c_x = centroid
    v_y, v_x = axis

    # Image dimensions
    height, width = image.shape

    # Extend the principal axis in both directions
    t = max(height, width)  # Extend the line far enough to span the image
    line_x = [c_x - t * v_x, c_x + t * v_x]
    line_y = [c_y - t * v_y, c_y + t * v_y]

    # Clip the line to the image boundaries
    line_x = np.clip(line_x, 0, width - 1)
    line_y = np.clip(line_y, 0, height - 1)

    # Plot the image
    plt.imshow(image, cmap="gray", origin="upper")
    plt.plot(line_x, line_y, color="red", linewidth=2, label="Principal Axis")
    plt.scatter(c_x, c_y, color="blue", label="Centroid", zorder=3)
    plt.title("Image with Principal Axis")
    plt.legend()
    plt.show()

