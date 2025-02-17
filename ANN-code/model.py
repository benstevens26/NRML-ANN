"""
model.py - Neural Network for Low Energy Nuclear Recoil Investigation (LENRI)
"""

import torch.nn as nn

class LENRI_CF4(nn.Module):
    def __init__(self, input_size=9, dropout_rate=0.0864):
        super(LENRI_CF4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(96, 62),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(62, 2)
        )
    
    def forward(self, x):
        return self.model(x)


class LENRI_Ar_CF4(nn.Module):
    def __init__(self, input_size=9, dropout_rate=0.0864):
        super(LENRI_Ar_CF4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 86),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(86, 48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(48, 3)
        )
    
    def forward(self, x):
        return self.model(x)

