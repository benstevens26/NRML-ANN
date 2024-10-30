import numpy as np
import scipy.ndimage as nd

def noise_adder():
    return None

def smooth_operator(image, smoothing_sigma=5):
    return nd.gaussian_filter(image, sigma=smoothing_sigma)

def extract_features(event, num_segments=50):
    """

    :param event: event object
    :return: array of features [length, energy, max_den, recoil_angle]
    """

    axis, mean_x, mean_y = event.get_principal_axis()

    recoil_angle = event.get_recoil_angle(principal_axis=axis)

    segment_distances, segment_intensities = event.get_intensity_profile(num_segments)
    length = event.get_track_length(num_segments, segment_distances, segment_intensities)
    energy = event.energy
    max_den = event.get_max_den()
    name = event.name

    return np.array([name, length, energy, max_den, recoil_angle])



def event_processor(events):

    noisy_images = noise_adder(events)

    for i in range(len(events)):
        events[i].image = smooth_operator(noisy_images[i])


def load_events(folder_path):
    """
    load all events in folder_path into Event objects
    :param folder_path: path to event folder
    :return: list of events
    """
    event_objects = []
    files = os.listdir(folder_path)

    for f in files:
        file_path = os.path.join(folder_path, f)
        image = np.load(file_path)
        event = Event(f, image)
        event_objects.append(event)

    return event_objects


def yield_events(path):
    """Generator function to load events in chunks from the given path."""
    # Implement logic to yield events from the large dataset in a memory-efficient way
    # For example, reading a large file line by line or loading files one at a time
    for event_file in event_files:  # Replace with actual file loading logic
        yield event_file






