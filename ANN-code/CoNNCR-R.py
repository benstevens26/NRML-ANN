import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
from bb_event import *


def load_all_bb_events(base_dirs: list):
    """
    Generator function to load and yield Event objects from .npy files within specified directories.

    Parameters
    ----------
    base_dirs : list of str
        List of base directory paths containing .npy event files.

    Yields
    ------
    Event
        An Event object for each .npy file found in the specified directories.
    """
    events = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            # Sort directories and files to ensure consistent order
            dirs.sort()  # Sort directories alphabetically
            files = sorted(
                f for f in files if f.endswith(".npy")
            )  # Sort and filter files for .npy

            for file in files:
                file_path = os.path.join(root, file)
                # Load the event data from the .npy file
                image = np.load(file_path)
                event = BB_Event(file, image)
                events.append(event)
    return events


data = load_all_bb_events(["/vols/lz/MIGDAL/sim_ims/C", "/vols/lz/MIGDAL/sim_ims/F"])
num_categories = 2 # Change to 3 if argon included

## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(559, 415,1))
net = base_model.output
net = tf.keras.layers.Flatten()(net)
net = tf.keras.layers.Dense(256, activation=tf.nn.relu)(net)
net = tf.keras.layers.Dropout(0.5)(net)
preds = tf.keras.layers.Dense(num_categories, activation=tf.nn.softmax)(net)
model = tf.keras.Model(base_model.input, preds)


freeze=False
# Freeze convolutional layers if needed
if freeze:
    for layer in model.layers[:-4]:
        layer.trainable = False


opt = tf.keras.optimizers.Adam(lr=1e-6, decay=0) # Default values from the paper I'm "leaning on".
loss = 'binary_crossentropy' if num_categories == 2 else 'categorical_crossentropy'

model.compile(loss=loss,
                optimizer=opt,
                metrics=['accuracy'])

# Setup checkpoint callback
ckpt_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/CNN_checkpoints"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                    save_weights_only=False,
                                                    period=1,
                                                    save_best_only=False,
                                                    monitor='val_loss')

# Setup TensorBoard callback
log_dir = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/CNN_checkpoints/logs"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir)


# Split into 70% train, 15% validation, 15% test
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - train_ratio, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio),
    random_state=42,
)


## Preprocessing input
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
X_val = preprocess_input(X_val)


epochs=10
batch_size=32 # No clue if this is applicable. Again just guessing based on the code I'm "inspired by"

train_start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    class_weight=None, # look into changing this, might be good to 
                    callbacks=[tb_callback, ckpt_callback])

train_end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

history_filename = os.path.join(log_dir, 'history.json')
info_filename = os.path.join(log_dir, 'info.txt')

with open(history_filename, 'w') as file:
    json.dump(history.history, file)

with open(info_filename, 'w') as file:
    file.write('***Training Info***\n')
    file.write('Training Start: {}'.format(train_start_time))
    file.write('Training End: {}\n'.format(train_end_time))
    file.write('Arguments:\n')
    for arg in sys.argv:
        file.write('\t{}\n'.format(arg))
