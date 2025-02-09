"""
model.py - Neural Network for Low Energy Nuclear Recoil Investigation (LENRI)
"""

import torch.nn as nn

class LENRI(nn.Module):
    def __init__(self, input_size=9):
        super(LENRI, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(8, 2),  
            nn.Softmax(dim=1) 
        )

    def forward(self, x):
        return self.model(x)
