""" Module for CNN preprocessing

contains methods for changing the event.image to be correct for use in the CNN models


"""

import cv2
import tensorflow as tf

def resize_pad_image_tf(event, target_size=(224, 224)):
    """
    Rescales the image (with padding) using tensorflow for CNN input
    """

    event.image = tf.image.resize_with_pad(event.image, target_size[0], target_size[1])

    return

def pad_image(event, target_size=(415, 559)):





def resize_event_image(event, res=None):
    """

    :param event: event object
    :param res: resolution desired, defaults to None (will resize to the maximum)
    :return:
    """

    if res is None:

        return
