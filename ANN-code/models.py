"""
model.py - Neural Network for Low Energy Nuclear Recoil Investigation (LENRI)
"""

import torch.nn as nn

class LENRI_CF4_1(nn.Module):
    """
    LENRI hyperparameter tuned model for CF4-1 dataset.
    """
    def __init__(self, input_size=9):
        super(LENRI_CF4_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0864),
            nn.Linear(96, 62),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0864),
            nn.Linear(62, 2),
        )

    def forward(self, x):
        return self.model(x)
    

class LENRI_Ar_CF4_1(nn.Module):
    """
    LENRI hyperparameter tuned model for Ar-CF4-1 dataset.
    """
    def __init__(self, input_size=9):
        super(LENRI_Ar_CF4_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 117),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0693),
            nn.Linear(117, 40),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0693),
            nn.Linear(40, 48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.0693),
            nn.Linear(48, 3),
        )

    def forward(self, x):
        return self.model(x)
    