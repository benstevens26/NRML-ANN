from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

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

## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=[559, 415])
base_model.trainable = False  ## Not trainable weights


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
