import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from cnn_processing import noise_adder, smooth_operator, pad_image_3

# Paths to your dataset
base_dirs = ["/vols/lz/MIGDAL/sim_ims/C", "/vols/lz/MIGDAL/sim_ims/F"]
carbon_dir = "/vols/lz/MIGDAL/sim_ims/C"
fluorine_dir = "/vols/lz/MIGDAL/sim_ims/F"
dark_list_number = 0
binning = 1
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"
m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
example_dark_list_unbinned = np.load(
    f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
)

# Function to load and preprocess a single image
def load_and_preprocess(filepath, label=None):
    filepath_str = filepath.numpy().decode("utf-8")

    # Load the .npy file
    image = np.load(filepath_str)

    # Extract label from file name ('C' or 'F')
    # label = (
    #     0 if "C" in os.path.basename(filepath_str) else 1
    # )
    
    # Preprocessing steps
    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list_unbinned)
    image = smooth_operator(image)
    image = pad_image_3(image)
    image = np.stack([image]*3, axis=-1)  # Duplicate channels
    image = preprocess_input(image)  # VGG16 preprocessing
    image = image / np.max(image)
    # image = tf.image.resize(image, (415, 559))  # Resize if needed
    
    return image, label

# Wrap the function for TensorFlow compatibility
def tf_load_and_preprocess(filepath, label):
    return tf.py_function(func=load_and_preprocess, inp=[filepath, label], Tout=(tf.float32, tf.int32))

# Create dataset from file paths
def create_dataset(file_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(tf_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)  # Batch size and prefetch for efficiency
    return dataset


file_list = []
labels = []
for base_dir in base_dirs:
    for root, dirs, files in os.walk(base_dir):
        files = [f for f in files if f.endswith(".npy")]
        file_list.extend([os.path.join(root, file) for file in files])
        labels.extend([0 if "C" in f else 1 for f in files if f.endswith(".npy")])

file_list.sort()
np.random.seed(77)
np.random.shuffle(file_list)

dataset = create_dataset(file_list,labels)