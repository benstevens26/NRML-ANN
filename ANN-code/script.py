"""
SSH SCRIPT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import os

from image_preprocessing import noise_adder, gaussian_smoothing
from data_methods import create_file_paths
from bb_event import Event

from feature_extraction import extract_sum_intensity, extract_max_intensity
from feature_extraction import extract_axis, extract_recoil_angle
from feature_extraction import extract_intensity_profile
from feature_extraction import extract_length

# feature extraction parameters
smoothing_sigma = 3.5
length_percentile = 40
local = False

# image directories
if local:
    base_dirs = ["ANN-code/Data/im0/C", "ANN-code/Data/im0/F"]
else:
    base_dirs = [
        "/vols/lz/tmarley/GEM_ITO/run/im0/C",
        "/vols/lz/tmarley/GEM_ITO/run/im0/F",
        "/vols/lz/tmarley/GEM_ITO/run/im1/C",
        "/vols/lz/tmarley/GEM_ITO/run/im1/F",
        "/vols/lz/tmarley/GEM_ITO/run/im2/C",
        "/vols/lz/tmarley/GEM_ITO/run/im2/F",
        "/vols/lz/tmarley/GEM_ITO/run/im3/C",
        "/vols/lz/tmarley/GEM_ITO/run/im3/F",
        "/vols/lz/tmarley/GEM_ITO/run/im4/C",
        "/vols/lz/tmarley/GEM_ITO/run/im4/F",
    ]


events = []
dir_number = 0

if local:
    dark_dir = "ANN-code/Data/darks"
else:
    dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"

for base_dir in base_dirs:

    # create list of image paths from a base directory
    image_paths = create_file_paths([base_dir])
    num_ims = len(image_paths)

    # removing known bad images if they are in the list
    if not local:
        uncropped_error = np.loadtxt("uncropped_error.csv", delimiter=",", dtype=str)
        min_dim_error = np.loadtxt("min_dim_error.csv", delimiter=",", dtype=str)
        errors = np.concatenate((uncropped_error, min_dim_error))
        image_paths = [path for path in image_paths if path not in errors]

        print(f"Removed {num_ims - len(image_paths)} known bad images")
        num_ims = len(image_paths)

    # event instantiation and preprocessing

    dark_list_number = dir_number
    m_dark = np.load(f"{dark_dir}/master_dark_1x1.npy")
    example_dark_list = np.load(f"{dark_dir}/quest_std_dark_{dark_list_number}.npy")

    dir_number += 1

    print("Preprocessing batch", dir_number)
    for path in tqdm(image_paths):
        # preprocessing images (add noise and smooth)
        im = np.load(path)
        im = noise_adder(im, m_dark, example_dark_list)
        im = gaussian_smoothing(im, smoothing_sigma=smoothing_sigma)

        events.append(Event(path, np.load(path)))

print("---------------------------------")
print("Preprocessing complete")
print("---------------------------------")

# Define columns for the features dataframe
features = [
    "file_name",
    "sum_intensity_camera",
    "max_intensity_camera",
    "recoil_angle_camera",
    "recoil_length_camera",
    "mean_energy_deposition_camera",
    "std_energy_deposition_camera",
    "skew_energy_deposition_camera",
    "kurt_energy_deposition_camera",
    "head_tail_mean_difference_camera",
]

features_dataframe = pd.DataFrame(columns=features)

# extract features for each events
print("Extracting features")
for event in tqdm(events):

    # basic features
    filename = os.path.basename(event.name)
    sum_intensity_camera = extract_sum_intensity(event.image)
    max_intensity_camera = extract_max_intensity(event.image)

    # axis features
    axis_camera, centroid_camera = extract_axis(event.image)
    recoil_angle_camera = extract_recoil_angle(axis_camera)

    # intensity profile features
    distances, intensities = extract_intensity_profile(
        event.image, principal_axis=axis_camera, centroid=centroid_camera
    )

    recoil_length_camera = extract_length(
        event.image, distances=distances, intensities=intensities
    )
    mean_energy_deposition_camera = np.mean(intensities)
    std_energy_deposition_camera = np.std(intensities)
    skew_energy_deposition_camera = sp.stats.skew(intensities)
    kurt_energy_deposition_camera = sp.stats.kurtosis(intensities)
    head_tail_mean_difference_camera = np.mean(
        intensities[: len(intensities) // 2]
    ) - np.mean(intensities[len(intensities) // 2 :])

    features_dataframe = features_dataframe._append(
        {
            "file_name": filename,
            "sum_intensity_camera": sum_intensity_camera,
            "max_intensity_camera": max_intensity_camera,
            "recoil_angle_camera": recoil_angle_camera,
            "recoil_length_camera": recoil_length_camera,
            "mean_energy_deposition_camera": mean_energy_deposition_camera,
            "std_energy_deposition_camera": std_energy_deposition_camera,
            "skew_energy_deposition_camera": skew_energy_deposition_camera,
            "kurt_energy_deposition_camera": kurt_energy_deposition_camera,
            "head_tail_mean_difference_camera": head_tail_mean_difference_camera,
        },
        ignore_index=True,
    )

print("---------------------------------")
print("Features extracted")
print("---------------------------------")

# save features to csv
features_dataframe.to_csv("features_raw.csv", index=False)
print("Features saved to csv")
