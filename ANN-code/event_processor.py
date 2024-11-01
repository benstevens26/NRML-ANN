import scipy.ndimage as nd
import csv
import os
from convert_sim_ims import *
from event import Event


def smooth_operator(event, smoothing_sigma=5):
    """
    Apply Gaussian smoothing to an event image.

    Parameters
    ----------
    event : Event
        The Event object containing the image to smooth.
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel, controlling the smoothing level (default is 5).

    Returns
    -------
    Event
        The Event object with the smoothed image.
    """
    event.image = nd.gaussian_filter(event.image, sigma=smoothing_sigma)

    return event


def noise_adder(event, m_dark=None, example_dark_list=None,noise_index=None):
    """
    Add noise to an event image based on a master dark image and random sample from example darks.

    Parameters
    ----------
    event : Event
        The Event object with an image to which noise will be added.
    m_dark : np.ndarray
        2D array (image) containing the master dark.
    example_dark_list : list of np.ndarray
        List of example dark images from which a random sample is selected for noise addition.

    Returns
    -------
    Event
        The Event object with noise added to the image.
    """

    if m_dark is None or example_dark_list is None:
        print("WARNING: Noise isn't being added.")
        return event

    if noise_index is None:
        noise_index = np.random.randint(0, len(example_dark_list) - 1)
    
    event.noise_index = noise_index
    event.image = convert_im(
        event.image,
        get_dark_sample(
            m_dark,
            [len(event.image[0]), len(event.image)],
            example_dark_list[noise_index],
        ),
    )
    return event


def extract_features(event, num_segments=15):
    """
    Extract key features from an event for classification.

    Parameters
    ----------
    event : Event
        The Event object to extract features from.
    num_segments : int, optional
        The number of segments for intensity profiling along the principal axis (default is 15).

    Returns
    -------
    np.ndarray
        Array of features including [name, length, energy, max_den, recoil_angle].
    """

    axis, mean_x, mean_y = event.get_principal_axis()

    recoil_angle = event.get_recoil_angle()

    segment_distances, segment_intensities = event.get_intensity_profile(num_segments)
    length = event.get_track_length(
        num_segments, segment_distances, segment_intensities
    )
    energy = event.get_track_intensity()
    max_den = event.get_max_den()
    name = event.name
    noise_index = event.noise_index

    return np.array([name, noise_index, length, energy, max_den, recoil_angle])


def event_processor(events, chunk_size, output_csv, m_dark, example_dark_list):
    """
    Process events in chunks and write extracted features to a CSV file.

    Parameters
    ----------
    events : generator
        A generator yielding Event objects to be processed.
    chunk_size : int
        Number of events to process and write at once.
    output_csv : str
        Path to the output CSV file where features will be saved.

    Writes
    ------
    CSV file
        Updates the output CSV file with feature data for each processed chunk.
    """
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Optionally write a header row if your feature extraction has fixed feature names
        writer.writerow(
            ["Name", "Noise Index", "Length", "Energy", "Max_Den", "Recoil Angle"]
        )  # Example headers

        chunk = []

        for event in events:
            event = noise_adder(event, m_dark, example_dark_list)
            event = smooth_operator(event)
            features = extract_features(event)
            chunk.append(features)

            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []

        if chunk:
            writer.writerows(chunk)


def yield_events(base_dirs):
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
