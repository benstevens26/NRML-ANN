""" Module for CNN preprocessing

contains methods for changing the event.image to be correct for use in the CNN models


"""

import os
import random

import cv2
import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
from bb_event import BB_Event
from convert_sim_ims import convert_im, get_dark_sample
from tensorflow.keras.applications.vgg16 import preprocess_input  # type: ignore


# Not checked either of these classes. They might just be chatgpt gobbledegook.
class SmoothOperator(tf.keras.layers.Layer):
    def __init__(self, smoothing_sigma=5, **kwargs):
        """
        A custom layer that applies a Gaussian smoothing filter to the input.
        """
        super(SmoothOperator, self).__init__(**kwargs)
        self.smoothing_sigma = smoothing_sigma

    def call(self, inputs):
        # This inner function will be run in Python.
        def _smooth(x):
            # x is a NumPy array.
            # Apply Gaussian filter from scipy.ndimage:
            return nd.gaussian_filter(x, sigma=self.smoothing_sigma)

        # Wrap the python function in tf.py_function.
        outputs = tf.py_function(func=_smooth, inp=[inputs], Tout=inputs.dtype)
        # Make sure the output tensor has the same shape as the input.
        outputs.set_shape(inputs.shape)
        return outputs


class NoiseAdder(tf.keras.layers.Layer):
    def __init__(self, m_dark=None, example_dark_list=None, **kwargs):
        """
        A custom layer that adds noise to the image.
        If m_dark or example_dark_list is not provided, a warning is printed and the image is returned unchanged.
        """
        super(NoiseAdder, self).__init__(**kwargs)
        self.m_dark = m_dark
        self.example_dark_list = example_dark_list

    def call(self, inputs):
        def _add_noise(x):
            # x is a NumPy array.
            if self.m_dark is None or self.example_dark_list is None:
                print("WARNING: Noise isn't being added.")
                return x
            # Shuffle the example_dark_list using TF and then convert to a NumPy value.
            shuffled = tf.random.shuffle(tf.convert_to_tensor(self.example_dark_list))
            sample_dark = shuffled[0].numpy()
            # Assume that get_dark_sample returns a dark sample given the parameters
            # and that convert_im applies that dark sample to the image.
            # We use x.shape to get dimensions (x.shape: (height, width, channels))
            dark_sample = get_dark_sample(
                self.m_dark, [x.shape[1], x.shape[0]], sample_dark
            )
            return convert_im(x, dark_sample)

        outputs = tf.py_function(func=_add_noise, inp=[inputs], Tout=inputs.dtype)
        outputs.set_shape(inputs.shape)
        return outputs


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
        target_frame[
            y_offset : y_offset + small_height, x_offset : x_offset + small_width
        ] = small_image

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
        target_frame[
            y_offset : y_offset + small_height, x_offset : x_offset + small_width
        ] = small_image

        return target_frame

    except:
        "Image could not fit inside target frame"


def pad_image_3(small_image, target_size=(559, 415)):

    # Convert the input numpy array to a TensorFlow tensor
    small_image = tf.convert_to_tensor(small_image, dtype=tf.float32)

    # Ensure the image has 3 dimensions (add a channel dimension if needed)
    if len(small_image.shape) == 2:
        small_image = tf.expand_dims(small_image, axis=-1)  # Add a channel dimension

    # Get the dimensions of the small image
    small_height, small_width = small_image.shape[:2]
    target_height, target_width = target_size

    # Calculate maximum offsets for random placement
    max_y_offset = target_height - small_height
    max_x_offset = target_width - small_width

    # Generate random offsets within the allowable range
    y_offset = random.randint(0, max_y_offset)
    x_offset = random.randint(0, max_x_offset)

    # Calculate the padding values
    top_padding = y_offset
    bottom_padding = max_y_offset - y_offset
    left_padding = x_offset
    right_padding = max_x_offset - x_offset

    # Use tf.image.pad_to_bounding_box for padding
    padded_image = tf.image.pad_to_bounding_box(
        small_image,
        offset_height=top_padding,
        offset_width=left_padding,
        target_height=target_height,
        target_width=target_width,
    )

    # Update the event image
    return padded_image.numpy()  # Convert back to numpy if needed


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
            # example_dark_list[np.random.randint(0, len(example_dark_list) - 1)],
            tf.random.shuffle(example_dark_list)[
                0
            ],  # Apparently tensorflow didn't like using numpy's randomiser. Not sure how to properly seed this randomisation beacuse it's tf not np. May need to come back here later
        ),
    )
    return image


def pad_image(image, target_size=(415, 559)):

    small_height, small_width = image.shape[:2]
    target_height, target_width = target_size

    # Create an empty frame filled with zeros (black) of size (415, 559)
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
    if type(file_path) == str:
        file_path_str = file_path
    else:
        try:
            file_path_str = file_path.numpy().decode("utf-8")
        except Exception as e:
            print(f"HANDLED FILE PATH BADLY IN PARSE_FUNCTION {e}")

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
    #     image = np.expand_dims(image, axis=-1)  # Shape becomes (415, 559, 1)

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
            image = image / max_val
        else:
            print("POTENTIAL ERROR: max value in image is 0 or less")
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


def parse_function_2(
    file_path, m_dark, example_dark_list_unbinned, channels=1, binning=1
):
    # Ensure `file_path` is a string (needed for `np.load`)
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")  # Convert from Tensor to string

    # Load the image
    image = np.load(file_path)

    # Extract label (assumes filenames contain "C" or "F")
    label = 0 if "C" in os.path.basename(file_path) else 1  # 0: Carbon, 1: Fluorine

    # Apply binning if necessary
    example_dark_list = (
        example_dark_list_unbinned
        if binning == 1
        else [bin_image(i, binning) for i in example_dark_list_unbinned]
    )

    # Apply processing functions
    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image = smooth_operator(image)
    image = pad_image(image)

    # Convert to float32
    image = image.astype(np.float32)

    # Ensure correct number of channels
    if channels == 3:
        # Normalize and stack channels
        max_val = np.max(image)
        if max_val > 0:
            image = image / max_val
        else:
            print("Warning: Image max value is 0, potential issue in normalization.")

        # Create 3-channel image
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

        # Apply VGG16 preprocessing
        image = preprocess_input(image)
    else:
        # Ensure grayscale format
        image = np.expand_dims(image, axis=-1)

    # Convert to TensorFlow tensors
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int32)

    return image, label


def parse_function_bb(file_path, channels=3, binning=1):
    # Ensure `file_path` is a string (needed for `np.load`)
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")  # Convert from Tensor to string

    # Load the image
    image = np.load(file_path)

    # Extract label (assumes filenames contain "C" or "F")
    label = 0 if "C" in os.path.basename(file_path) else 1  # 0: Carbon, 1: Fluorine

    # # Apply binning if necessary
    # example_dark_list = (
    #     example_dark_list_unbinned
    #     if binning == 1
    #     else [bin_image(i, binning) for i in example_dark_list_unbinned]
    # )

    # # Apply processing functions
    # image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    # image = smooth_operator(image)
    # image = pad_image(image)

    # Convert to float32
    image = image.astype(np.float32)

    # Ensure correct number of channels
    if channels == 3:
        # Normalize and stack channels
        max_val = np.max(image)
        if max_val > 0:
            image = image / max_val
        else:
            print("Warning: Image max value is 0, potential issue in normalization.")

        # Create 3-channel image
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

        # Apply VGG16 preprocessing
        image = preprocess_input(image)
    else:
        # Ensure grayscale format
        image = np.expand_dims(image, axis=-1)

    # Convert to TensorFlow tensors
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int32)

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
            tf.ensure_shape(image, (415, 559, channels)),
            tf.ensure_shape(label, ()),
        )
    )

    # Shuffle, batch, and prefetch the data for training
    # dataset = dataset.shuffle(buffer_size=100)  # Shuffle the dataset to ensure randomness
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def load_data_yield(base_dirs, example_dark_tensor, m_dark_tensor, channels=1):
    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.endswith(".npy")]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Process the single image
    for file_path in file_list:
        image, label = parse_function_2(
            file_path, m_dark_tensor.numpy(), example_dark_tensor.numpy(), channels
        )
        yield image, label

    # # Set output shapes explicitly to avoid unknown rank issues
    # dataset = dataset.map(
    #     lambda image, label: (
    #         tf.ensure_shape(image, (415, 559, channels)),
    #         tf.ensure_shape(label, ()),
    #     )
    # )


def load_data_yield_bb(base_dirs, channels=3):
    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.endswith(".npy")]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Process the single image
    for file_path in file_list:
        image, label = parse_function_bb(file_path, channels)
        yield image, label
