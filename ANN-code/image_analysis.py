"""
Module that contains standalone functions for imaging analysis.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_axis(image, axis, centroid):
    """
    Plots the principal axis on an image.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        principal_axis (numpy.ndarray): Unit vector indicating the direction of the axis.
        centroid (tuple): Centroid coordinates (y, x).
    """
    height, width = image.shape
    c_y, c_x = centroid
    v_y, v_x = axis

    # Calculate intersection points with image boundaries
    # Solve for y = m*x + b, extended to all boundaries
    if v_x != 0:  # Avoid division by zero for vertical lines
        slope = v_y / v_x
        # Intersect with left (x=0) and right (x=width) boundaries
        y0 = c_y - slope * c_x
        y_width = c_y + slope * (width - c_x)
    else:  # Vertical line
        y0, y_width = 0, height

    if v_y != 0:  # Avoid division by zero for horizontal lines
        inv_slope = v_x / v_y
        # Intersect with top (y=0) and bottom (y=height) boundaries
        x0 = c_x - inv_slope * c_y
        x_height = c_x + inv_slope * (height - c_y)
    else:  # Horizontal line
        x0, x_height = 0, width

    # Clip the line to image boundaries
    line_x = np.clip([x0, x_height], 0, width)
    line_y = np.clip([y0, y_width], 0, height)

    # Plot the image and principal axis
    plt.imshow(image, cmap="viridis")
    plt.plot(line_x, line_y, color="red", label="Principal Axis")
    plt.scatter(c_x, c_y, color="blue", label="Centroid")
    plt.title("Principal Axis on Image")
    plt.legend()
    plt.show()