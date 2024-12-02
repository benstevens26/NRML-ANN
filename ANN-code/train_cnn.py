import tensorflow as tf
import numpy as np
import os
import random
import scipy.ndimage as nd
from convert_sim_ims import convert_im, get_dark_sample
import pickle

from cnn_processing import bin_image, smooth_operator, noise_adder, pad_image, parse_function, load_data

# Define base directories and batch size

# base_dirs = ['/vols/lz/MIGDAL/sim_ims/C', '/vols/lz/MIGDAL/sim_ims/F']  # List your data directories here
base_dirs = ['Data/C', 'Data/F']  # List your data directories here
batch_size = 32
dark_list_number = 0
binning = 1
# dark_dir="/vols/lz/MIGDAL/sim_ims/darks"
dark_dir="Data/darks"
m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
example_dark_list_unbinned = np.load(
    f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
)

# Load the dataset
full_dataset = load_data(base_dirs, batch_size, example_dark_list_unbinned, m_dark)

dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size  # Ensure all data is used

train_dataset = full_dataset.take(train_size)  # First 70%
remaining = full_dataset.skip(train_size)  # Remaining 30%
val_dataset = remaining.take(val_size)  # Next 15%
test_dataset = remaining.skip(val_size)  # Final 15%

# Define the model
# CoNNCR = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(415, 559, 1)),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

CoNNCR = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(415, 559, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
CoNNCR.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(CoNNCR.summary())
print(tf.config.list_physical_devices())

exit()

# Train the model
CoNNCR.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Save the trained model
CoNNCR.save('CoNNCR.keras')

# Save model training history
history_save_path = "CoNNCR_history.pkl"
with open(history_save_path, "wb") as file:
    pickle.dump(CoNNCR.history, file)
