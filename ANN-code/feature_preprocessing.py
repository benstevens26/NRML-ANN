"""
feature_preprocessing.py - Data loading and preprocessing for LENRI classification

This script handles:
- Reading the dataset from CSV
- Extracting numerical features
- Extracting labels (C -> 0, F -> 1) from the 'file_name' column
- Splitting into train, validation, and test sets (70% train, 15% val, 15% test)
- Ensuring training and validation sets have an equal number of C and F
- Ensuring the test set has 80% fluorine and 20% carbon
- Creating PyTorch Datasets and DataLoaders
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re

def extract_label(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

class NuclearRecoilDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.iloc[:, 1:].values  # Extract numerical features
        self.labels = dataframe["label"].values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_dataloaders(csv_file, batch_size=32):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label)

    # Drop rows with NaN labels (if any)
    df = df.dropna(subset=["label"])
    
    # Split data into 70% train, 15% validation, 15% test
    train_val, test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val["label"], random_state=42)  # 0.1765 ensures 15% of original dataset
    
    # Ensure training and validation sets are balanced (equal C and F)
    min_train_samples = min(train[train["label"] == 0].shape[0], train[train["label"] == 1].shape[0])
    train_c = train[train["label"] == 0].sample(n=min_train_samples, random_state=42)
    train_f = train[train["label"] == 1].sample(n=min_train_samples, random_state=42)
    train = pd.concat([train_c, train_f]).sample(frac=1, random_state=42)  # Shuffle
    
    min_val_samples = min(val[val["label"] == 0].shape[0], val[val["label"] == 1].shape[0])
    val_c = val[val["label"] == 0].sample(n=min_val_samples, random_state=42)
    val_f = val[val["label"] == 1].sample(n=min_val_samples, random_state=42)
    val = pd.concat([val_c, val_f]).sample(frac=1, random_state=42)  # Shuffle
    
    # Ensure test set follows 80% F, 20% C ratio
    num_f = int(0.8 * len(test))
    num_c = len(test) - num_f
    test_f = test[test["label"] == 1].sample(n=num_f, random_state=42)
    test_c = test[test["label"] == 0].sample(n=num_c, random_state=42)
    test = pd.concat([test_f, test_c]).sample(frac=1, random_state=42)  # Shuffle
    
    # Convert to PyTorch datasets
    train_dataset = NuclearRecoilDataset(train)
    val_dataset = NuclearRecoilDataset(val)
    test_dataset = NuclearRecoilDataset(test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
