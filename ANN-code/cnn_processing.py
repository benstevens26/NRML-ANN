""" Module for CNN preprocessing

contains methods for changing the event.image to be correct for use in the CNN models


"""

import cv2
import tensorflow as tf
import random
import numpy as np


def resize_pad_image_tf(event, target_size=(224, 224)):
    """
    Rescales the image (with padding) using tensorflow for CNN input
    """

    event.image = tf.image.resize_with_pad(event.image, target_size[0], target_size[1])


def pad_image(event, target_size=(415, 559)):

    small_image = event.image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (415, 559)
        target_frame = np.zeros((target_height, target_width), dtype=small_image.dtype)

        # Calculate maximum offsets so the small image fits inside the target frame
        max_y_offset = target_height - small_height
        max_x_offset = target_width - small_width

        # Generate random offsets within the allowable range
        y_offset = random.randint(0, max_y_offset)
        x_offset = random.randint(0, max_x_offset)

        # Insert the small image into the target frame at the random offset
        target_frame[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = small_image

        event.image = target_frame

    except:
        "Image could not fit inside target frame"



def pad_image_2(image, target_size=(415, 559)):

    small_image = image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (415, 559)
        target_frame = np.zeros((target_height, target_width), dtype=small_image.dtype)

        # Calculate maximum offsets so the small image fits inside the target frame
        max_y_offset = target_height - small_height
        max_x_offset = target_width - small_width

        # Generate random offsets within the allowable range
        y_offset = random.randint(0, max_y_offset)
        x_offset = random.randint(0, max_x_offset)

        # Insert the small image into the target frame at the random offset
        target_frame[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = small_image

        return target_frame

    except:
        "Image could not fit inside target frame"

