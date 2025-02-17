#!/usr/bin/env python3
"""
SSH SCRIPT
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from feature_preprocessing import get_dataloaders_cf4, get_dataloaders_ar_cf4
from model import LENRI_CF4, LENRI_Ar_CF4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "LENRI_CF4_old_unopt.pth"
features_path = "ANN-code/Data/features_old/all_features_2_scaled.csv"
model = LENRI_CF4().to(device)
binary = True

# Load test set
if binary:
    _, _, test_loader = get_dataloaders_cf4(features_path, batch_size=32)
else:
    _, _, test_loader = get_dataloaders_ar_cf4(features_path, batch_size=32)


checkpoint = torch.load(model_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())  # Debugging
