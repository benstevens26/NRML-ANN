""" Module for CNN preprocessing

contains methods for changing the event.image to be correct for use in the CNN models


"""

import cv2
import tensorflow as tf
import random
import numpy as np
from bb_event import BB_Event
import os
import random
import scipy.ndimage as nd
from convert_sim_ims import convert_im, get_dark_sample
from tensorflow.keras.applications.vgg16 import preprocess_input


def resize_pad_image_tf(event, target_size=(224, 224)):
    """
    Rescales the image (with padding) using tensorflow for CNN input
    """

    event.image = tf.image.resize_with_pad(event.image, target_size[0], target_size[1])


def pad_image(event, target_size=(768, 768)):

    small_image = event.image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (768, 768)
        target_frame = np.zeros((target_height, target_width), dtype=small_image.dtype)

        # Calculate maximum offsets so the small image fits inside the target frame
        max_y_offset = target_height - small_height
        max_x_offset = target_width - small_width

        # Generate random offsets within the allowable range
        y_offset = random.randint(0, max_y_offset)
        x_offset = random.randint(0, max_x_offset)

        # Insert the small image into the target frame at the random offset
        target_frame[
            y_offset : y_offset + small_height, x_offset : x_offset + small_width
        ] = small_image

        event.image = target_frame

    except:
        "Image could not fit inside target frame"


def pad_image_2(image, target_size=(768, 768)):

    small_image = image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (768, 768)
        target_frame = np.zeros((target_height, target_width), dtype=small_image.dtype)

        # Calculate maximum offsets so the small image fits inside the target frame
        max_y_offset = target_height - small_height
        max_x_offset = target_width - small_width

        # Generate random offsets within the allowable range
        y_offset = random.randint(0, max_y_offset)
        x_offset = random.randint(0, max_x_offset)

        # Insert the small image into the target frame at the random offset
        target_frame[
            y_offset : y_offset + small_height, x_offset : x_offset + small_width
        ] = small_image

        return target_frame

    except:
        "Image could not fit inside target frame"


def load_events_bb(file_path):
    """
    load events from folder
    """

    events = []

    # Iterate over all .npy files in the directory
    for filename in os.listdir(file_path):
        if filename.endswith(".npy"):
            # Construct full file path
            full_path = os.path.join(file_path, filename)

            # Load image data from .npy file
            image = np.load(full_path)

            # Instantiate Event object
            event = BB_Event(filename, image)

            # Append to events list
            events.append(event)

    return events


def bin_image(image, N: int):

    height, width = image.shape

    new_height = (height // N) * N
    new_width = (width // N) * N

    trimmed_image = image[:new_height, :new_width]

    if N > 1:
        binned_image = trimmed_image.reshape(new_height // N, N, new_width // N, N).sum(
            axis=(1, 3)
        )
        return binned_image
    else:
        return trimmed_image


def smooth_operator(image, smoothing_sigma=5):

    image = nd.gaussian_filter(image, sigma=smoothing_sigma)

    return image


def noise_adder(image, m_dark=None, example_dark_list=None):

    if m_dark is None or example_dark_list is None:
        print("WARNING: Noise isn't being added.")
        return image

    image = convert_im(
        image,
        get_dark_sample(
            m_dark,
            [len(image[0]), len(image)],
            example_dark_list[np.random.randint(0, len(example_dark_list) - 1)],
        ),
    )
    return image


def pad_image(image, target_size=(768, 768)):

    small_height, small_width = image.shape[:2]
    target_height, target_width = target_size

    # Create an empty frame filled with zeros (black) of size (768, 768)
    target_frame = np.zeros((target_height, target_width), dtype=image.dtype)

    # Calculate maximum offsets so the small image fits inside the target frame
    max_y_offset = target_height - small_height
    max_x_offset = target_width - small_width

    # Generate random offsets within the allowable range
    y_offset = random.randint(0, max_y_offset)
    x_offset = random.randint(0, max_x_offset)

    # Insert the small image into the target frame at the random offset
    target_frame[
        y_offset : y_offset + small_height, x_offset : x_offset + small_width
    ] = image

    return target_frame


# Function to load a single file and preprocess it
# def parse_function(file_path, binning=1, dark_dir="/vols/lz/MIGDAL/sim_ims/darks"):
def parse_function(
    file_path, m_dark, example_dark_list_unbinned, channels=1, binning=1
):

    file_path_str = file_path.numpy().decode("utf-8")

    # Load the image data from the .npy file
    image = np.load(file_path_str)

    # Extract label from file name ('C' or 'F')
    label = (
        0 if "C" in os.path.basename(file_path_str) else 1
    )  # Assume 'C' maps to 0 and 'F' maps to 1

    if binning != 1:
        example_dark_list = [bin_image(i, binning) for i in example_dark_list_unbinned]

    else:
        example_dark_list = example_dark_list_unbinned

    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image = smooth_operator(image)
    image = pad_image(image)
    # print(image.shape)
    # Set shape explicitly for TensorFlow to know
    image = np.float32(image)
    # label = np.int32(label)
    # if channels == 3:
    # # Ensure float type and normalize
    #     image = image.astype(np.float32)
    #     max_val = np.max(image)
    #     if max_val > 0:  # Avoid division by zero
    #         image /= max_val
    #     image = np.stack([image]*channels, axis=-1)
    #     # image = preprocess_input(image)

    # else:
    #     image = np.expand_dims(image, axis=-1)  # Shape becomes (768, 768, 1)

    if channels == 3:
        # Ensure the image is a 2D array
        if image.ndim == 3:
            if image.shape[-1] == 1:  # Handle (height, width, 1)
                image = np.squeeze(image, axis=-1)
            else:
                raise ValueError(
                    f"Unexpected shape {image.shape} for single-channel image."
                )
        elif image.ndim != 2:
            raise ValueError(
                f"Unexpected image shape: {image.shape}. Expected 2D array."
            )

        # Ensure the type is float32
        image = image.astype(np.float32)

        # Normalize to range [0, 255] if needed
        max_val = np.max(image)
        if max_val > 0:  # Avoid division by zero
            image = image * (255.0 / max_val)

        # Create 3 channels by stacking
        image = np.repeat(
            image[:, :, np.newaxis], 3, axis=-1
        )  # Shape becomes (height, width, 3)

        # Apply VGG16 preprocessing
        image = preprocess_input(image)
    else:
        # Expand dimensions for single-channel images
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Shape becomes (height, width, 1)
        elif image.ndim != 3 or image.shape[-1] != 1:
            raise ValueError(f"Unexpected shape {image.shape} for grayscale image.")

    # print(image.shape)
    # print(type(label))

    # label = np.int32(label)
    # image = np.float32(image)
    return image, label


# Dataset Preparation Function Using `tf.data`
def load_data(base_dirs, batch_size, example_dark_list, m_dark, channels=1):
    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.endswith(".npy")]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Create a TensorFlow dataset from the list of file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
    example_dark_tensor = tf.convert_to_tensor(example_dark_list, dtype=tf.float32)
    # Apply the parsing function
    dataset = dataset.map(
        lambda file_path: tf.py_function(
            func=parse_function,
            inp=[file_path, m_dark_tensor, example_dark_tensor, channels],
            Tout=(tf.float32, tf.int32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Set output shapes explicitly to avoid unknown rank issues
    dataset = dataset.map(
        lambda image, label: (
            tf.ensure_shape(image, (768, 768, channels)),
            tf.ensure_shape(label, ()),
        )
    )

    # Shuffle, batch, and prefetch the data for training
    # dataset = dataset.shuffle(buffer_size=100)  # Shuffle the dataset to ensure randomness
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
