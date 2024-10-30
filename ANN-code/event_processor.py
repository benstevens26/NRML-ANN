import numpy as np
import scipy.ndimage as nd
import os
from event import Event
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


def yield_events(base_dirs):
    """Generator function to load events in chunks from the given directories"""

    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            # Sort directories and files to ensure consistent order
            dirs.sort()  # Sort directories alphabetically
            files = sorted(f for f in files if f.endswith('.npy'))  # Sort and filter files for .npy

            for file in files:
                file_path = os.path.join(root, file)
                # Load the event data from the .npy file
                image = np.load(file_path)
                event = Event(file, image)

                yield event







