"""
feature_preprocessing.py - Data loading and preprocessing for LENRI classification

This script handles:
- Reading the dataset from CSV
- Extracting numerical features
- Extracting labels (C -> 0, F -> 1) from the 'file_name' column
- Splitting into train, validation, and test sets
- Ensuring training and validation sets have an equal number of C and F
- Creating PyTorch Datasets and DataLoaders
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class NuclearRecoilDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.iloc[:, 1:].values  # Extract numerical features
        self.labels = dataframe["file_name"].apply(lambda x: 0 if "C" in x else 1).values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_dataloaders(csv_file, batch_size=32, test_ratio=0.2, val_ratio=0.1):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(lambda x: 0 if "C" in x else 1)
    
    # Split into train+val and test sets
    train_val, test = train_test_split(df, test_size=test_ratio, stratify=df["label"], random_state=42)
    
    # Ensure test set follows the 80% Fluorine, 20% Carbon ratio
    test_f = test[test["label"] == 1].sample(frac=0.8, random_state=42)
    test_c = test[test["label"] == 0].sample(frac=0.2, random_state=42)
    test = pd.concat([test_f, test_c])
    
    # Ensure training and validation sets have equal numbers of C and F
    train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val["label"], random_state=42)
    num_c = train[train["label"] == 0].shape[0]
    num_f = train[train["label"] == 1].shape[0]
    min_samples = min(num_c, num_f)
    train_c = train[train["label"] == 0].sample(n=min_samples, random_state=42)
    train_f = train[train["label"] == 1].sample(n=min_samples, random_state=42)
    train = pd.concat([train_c, train_f])
    
    # Convert to PyTorch datasets
    train_dataset = NuclearRecoilDataset(train)
    val_dataset = NuclearRecoilDataset(val)
    test_dataset = NuclearRecoilDataset(test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
