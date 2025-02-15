"""
model.py - Neural Network for Low Energy Nuclear Recoil Investigation (LENRI)
"""

import torch.nn as nn

class LENRI_CF4(nn.Module):
    def __init__(self, input_size=9, dropout_rate=0.106):
        super(LENRI_CF4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 65),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(65, 71),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(71, 2)
        )
    
    def forward(self, x):
        return self.model(x)
