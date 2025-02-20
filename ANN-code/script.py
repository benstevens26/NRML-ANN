#!/usr/bin/env python3
"""
SSH SCRIPT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import os
import sys

from image_preprocessing import noise_adder, gaussian_smoothing
from bb_event import Event

from feature_extraction import extract_sum_intensity, extract_max_intensity
from feature_extraction import extract_axis, extract_recoil_angle
from feature_extraction import extract_intensity_profile
from feature_extraction import extract_length

# Get the job number from the argument (HTCondor passes $(PROCESS))
if len(sys.argv) > 1:
    job_number = int(sys.argv[1])  # Process ID from HTCondor
else:
    raise ValueError("No job number provided. Ensure the script is run via HTCondor.")

print(f"Running job {job_number}")

# feature extraction parameters CHANGE CHANGE CHANGE DONT FORGET CHANGE !!!
smoothing_sigma = 3.5
length_percentile = 40
num_jobs = 50
name = "Ar_CF4_2_"+str(job_number)
matched_files = "/vols/lz/bstevens/NR-ANN/ANN-code/matched_file_paths_Ar_CF4.csv"
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"

# Load matched file paths
df_matched = pd.read_csv(matched_files)

# Split files among jobs
total_files = len(df_matched)
files_per_job = total_files // num_jobs
start_idx = job_number * files_per_job
end_idx = total_files if job_number == num_jobs - 1 else (job_number + 1) * files_per_job

df_job = df_matched.iloc[start_idx:end_idx].reset_index(drop=True)

print(f"Processing {len(df_job)} files in job {job_number}")

# Make event objects with images preprocessed

file_paths = np.array(df_job)
events = []
dark_list_number = np.random.randint(0, 10)
m_dark = np.load(f"{dark_dir}/master_dark_1x1.npy")
example_dark_list = np.load(f"{dark_dir}/quest_std_dark_{dark_list_number}.npy")

print("---------------------------------")
print("Instantiating events and preprocessing images")
print("---------------------------------")

for cam_file, ito_file in file_paths:

    cam_im = np.load(cam_file)
    ito_im = np.load(ito_file)

    cam_im = noise_adder(cam_im, m_dark, example_dark_list)
    cam_im = gaussian_smoothing(cam_im, smoothing_sigma=smoothing_sigma)
    ito_im = gaussian_smoothing(ito_im, smoothing_sigma=smoothing_sigma)

    event = Event(cam_file, cam_im)
    event.ito_image = ito_im
    events.append(event)

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
    "sum_intensity_ito",
    "max_intensity_ito",
    "recoil_angle_ito",
    "recoil_length_ito",
    "mean_energy_deposition_ito",
    "std_energy_deposition_ito",
    "skew_energy_deposition_ito",
    "kurt_energy_deposition_ito",
    "head_tail_mean_difference_ito",
]

features_dataframe = pd.DataFrame(columns=features)

# Extract the features 
print("---------------------------------")
print("Starting feature extraction")
print("---------------------------------")

for event in tqdm(events):

    # BASIC FEATURES
    filename = event.name
    sum_intensity_camera = extract_sum_intensity(event.image)
    max_intensity_camera = extract_max_intensity(event.image)
    sum_intensity_ito = extract_sum_intensity(event.ito_image)
    max_intensity_ito = extract_max_intensity(event.ito_image)

    # PRINCIPAL AXES
    axis_camera, centroid_camera = extract_axis(event.image)
    axis_ito, centroid_ito = extract_axis(event.ito_image)
    recoil_angle_camera = extract_recoil_angle(axis_camera)
    recoil_angle_ito = extract_recoil_angle(axis_ito)

    # INTENSITY PROFILE FEATURES
    distances, intensities = extract_intensity_profile(
        event.image, principal_axis=axis_camera, centroid=centroid_camera
    )

    distances_ito, intensities_ito = extract_intensity_profile(
        event.ito_image, principal_axis=axis_ito, centroid=centroid_ito
    )

    if len(intensities) == 0 or intensities is None:
        print(f"Error extracting intensity profile for {filename}")
        continue

    if len(intensities_ito) == 0 or intensities_ito is None:
        print(f"Error extracting intensity profile for {filename}")
        continue

    recoil_length_camera = extract_length(
        event.image, distances=distances, intensities=intensities
    )

    recoil_length_ito = extract_length(
        event.ito_image, distances=distances_ito, intensities=intensities_ito
    )

    mean_energy_deposition_camera = np.mean(intensities)
    std_energy_deposition_camera = np.std(intensities)
    skew_energy_deposition_camera = sp.stats.skew(intensities)
    kurt_energy_deposition_camera = sp.stats.kurtosis(intensities)
    head_tail_mean_difference_camera = np.mean(
        intensities[: len(intensities) // 2]
    ) - np.mean(intensities[len(intensities) // 2 :])

    mean_energy_deposition_ito = np.mean(intensities_ito)
    std_energy_deposition_ito = np.std(intensities_ito)
    skew_energy_deposition_ito = sp.stats.skew(intensities_ito)
    kurt_energy_deposition_ito = sp.stats.kurtosis(intensities_ito)
    head_tail_mean_difference_ito = np.mean(
        intensities_ito[: len(intensities_ito) // 2]
    ) - np.mean(intensities_ito[len(intensities_ito) // 2 :])

    # Append features to dataframe
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
            "sum_intensity_ito": sum_intensity_ito,
            "max_intensity_ito": max_intensity_ito,
            "recoil_angle_ito": recoil_angle_ito,
            "recoil_length_ito": recoil_length_ito,
            "mean_energy_deposition_ito": mean_energy_deposition_ito,
            "std_energy_deposition_ito": std_energy_deposition_ito,
            "skew_energy_deposition_ito": skew_energy_deposition_ito,
            "kurt_energy_deposition_ito": kurt_energy_deposition_ito,
            "head_tail_mean_difference_ito": head_tail_mean_difference_ito,
        },
        ignore_index=True,
    )

print("---------------------------------")
print("Features extracted, saving to csv")
print("---------------------------------")

features_dataframe.to_csv("features_"+name+".csv", index=False)

print("---------------------------------")
print("Features saved to csv")
print("---------------------------------")