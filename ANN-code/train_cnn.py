import tensorflow as tf
import numpy as np
import os
import random
import scipy.ndimage as nd
from convert_sim_ims import convert_im, get_dark_sample
import pickle


def bin_image(image, N):

    height, width = image.shape

    new_height = (height // N) * N
    new_width = (width // N) * N

    trimmed_image = image[:new_height, :new_width]

    binned_image = trimmed_image.reshape(new_height // N, N, new_width // N, N).sum(
        axis=(1, 3)
    )

    return binned_image

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
            example_dark_list[np.random.randint(0, len(example_dark_list)-1)],
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
    target_frame[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = image

    return target_frame

# Function to load a single file and preprocess it
def parse_function(file_path, binning=1, dark_dir="/vols/lz/MIGDAL/sim_ims/darks"):

    file_path_str = file_path.numpy().decode('utf-8')

    # Load the image data from the .npy file
    image = np.load(file_path_str)

    # Extract label from file name ('C' or 'F')
    label = 0 if 'C' in os.path.basename(file_path_str) else 1  # Assume 'C' maps to 0 and 'F' maps to 1

    # Add noise and smooth the image
    dark_list_number = 0
    m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
    example_dark_list_unbinned = np.load(
        f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
    )
    example_dark_list = [
        bin_image(i, binning) for i in example_dark_list_unbinned
    ]

    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)

    # Pad the image
    image = pad_image(image)

    # Set shape explicitly for TensorFlow to know
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)  # Shape becomes (415, 559, 1)

    return image, label

# Dataset Preparation Function Using `tf.data`
def load_data(base_dirs, batch_size):
    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            dirs.sort()  # Ensure consistent directory traversal order
            files = sorted(f for f in files if f.endswith(".npy"))
            file_list.extend([os.path.join(root, file) for file in files])

    # Create a TensorFlow dataset from the list of file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    # Apply the parsing function
    dataset = dataset.map(lambda file_path: tf.py_function(func=parse_function,
                                                           inp=[file_path],
                                                           Tout=(tf.float32, tf.int32)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Set output shapes explicitly to avoid unknown rank issues
    dataset = dataset.map(lambda image, label: (
        tf.ensure_shape(image, (415, 559, 1)),
        tf.ensure_shape(label, ())
    ))

    # Shuffle, batch, and prefetch the data for training
    dataset = dataset.shuffle(buffer_size=10000)  # Shuffle the dataset to ensure randomness
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

# Define base directories and batch size
base_dirs = ['/vols/lz/MIGDAL/sim_ims/C', '/vols/lz/MIGDAL/sim_ims/F']  # List your data directories here
batch_size = 32

# Load the dataset
full_dataset = load_data(base_dirs, batch_size)

dataset_size = len(full_dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = int(0.15 * dataset_size)

train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)


# Define the model
CoNNCR = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(415, 559, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
CoNNCR.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
CoNNCR.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Save the trained model
CoNNCR.save('CoNNCR.keras')

# Save model training history
history_save_path = "CoNNCR_history.pkl"
with open(history_save_path, "wb") as file:
    pickle.dump(CoNNCR.history, file)
