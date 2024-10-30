"""
"""
import numpy as np

def extract_features(event, num_segments):
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

    return np.array([length, energy, max_den, recoil_angle])

