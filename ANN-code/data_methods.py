"""
Module for data management
"""

from tqdm import tqdm
import os
import numpy as np
import pandas as pd

def create_file_paths(base_dirs):
    """
    Create a list of file paths from a list of directories.

    Parameters:
    ----------
    base_dirs : list
        A list of directories containing .npy (image) files.

    Returns:
    -------
    file_paths : list
        A (randomised) list of file paths to the .npy files in the specified directories.

    """

    file_paths = []

    if not isinstance(base_dirs, list):
        base_dirs = [base_dirs]

    for dir in base_dirs:
        for filename in tqdm(os.listdir(dir)):
            if filename.endswith(".npy"):

                file_paths.append(os.path.join(dir, filename))

    np.random.shuffle(file_paths)

    return file_paths


def get_file_paths(base_dirs):
    """
    Retrieves all .npy file paths from given directories.
    
    Parameters:
        base_dirs (list of str): List of base directories to search for .npy files.
    
    Returns:
        dict: Dictionary mapping filenames (without suffix) to their full paths.
    """
    file_dict = {}
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            for file in os.listdir(base_dir):
                if file.endswith(".npy"):
                    filename_key = "_".join(file.split("_")[:-1])  # Remove suffix (im/ito.npy)
                    file_dict[filename_key] = os.path.join(base_dir, file)
    
    return file_dict

def match_files(cam_dirs, ito_dirs):
    """
    Matches .npy files between camera and ITO directories and stores in a dataframe.
    
    Parameters:
        cam_dirs (list of str): List of camera directories.
        ito_dirs (list of str): List of ITO directories.
    
    Returns:
        pd.DataFrame: Dataframe containing matched file paths.
    """
    cam_files = get_file_paths(cam_dirs)
    ito_files = get_file_paths(ito_dirs)
    
    matched_data = []
    for key in cam_files:
        if key in ito_files:
            matched_data.append((cam_files[key], ito_files[key]))
    
    df = pd.DataFrame(matched_data, columns=["cam_file_path", "ito_file_path"])
    return df
