import numpy as np
import scipy.ndimage as nd
import os
import csv
from tqdm import tqdm
from event import Event


def noise_adder(event):
    return event


def smooth_operator(event, smoothing_sigma=5):
    event.image = nd.gaussian_filter(event.image, sigma=smoothing_sigma)

    return event


def extract_features(event, num_segments=10):
    """

    :param event: event object
    :return: array of features [length, energy, max_den, recoil_angle]
    """

    axis, mean_x, mean_y = event.get_principal_axis()

    recoil_angle = event.get_recoil_angle(principal_axis=axis)

    segment_distances, segment_intensities = event.get_intensity_profile(num_segments)
    length = event.get_track_length(
        num_segments, segment_distances, segment_intensities
    )
    energy = event.get_track_energy()
    max_den = event.get_max_den()
    name = event.name

    return np.array([name, length, energy, max_den, recoil_angle])


def event_processor(events, chunk_size, output_csv):
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Optionally write a header row if your feature extraction has fixed feature names
        writer.writerow(
            ["Name", "Length", "Energy", "Max_Den", "Recoil Angle"]
        )  # Example headers

        chunk = []

        for event in tqdm(events):
            event = noise_adder(event)
            event = smooth_operator(event)
            features = extract_features(event)
            chunk.append(features)

            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []

        if chunk:
            writer.writerows(chunk)


def yield_events(base_dirs):
    """Generator function to load events in chunks from the given directories"""

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
                event = Event(file, image)

                yield event
