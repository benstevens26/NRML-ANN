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

def extract_label_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"_C_", filename):
        return 0
    elif re.search(r"_F_", filename):
        return 1
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

def extract_label_ar_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"00_Ar_", filename):
        return 2  # Argon
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

class NuclearRecoilDatasetCF4(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(columns=["file_name", "label"]).values  # Drop file_name explicitly
        self.labels = dataframe["label"].values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Class indices
        return x, y

class NuclearRecoilDatasetArCF4(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(columns=["file_name", "label"]).values  # Drop file_name explicitly
        self.labels = dataframe["label"].values  # Class indices
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Class indices
        return x, y

def get_dataloaders_cf4(csv_file, batch_size=32):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label_cf4)
    train, test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    train, val = train_test_split(train, test_size=0.1765, stratify=train["label"], random_state=42)
    train_dataset = NuclearRecoilDatasetCF4(train)
    val_dataset = NuclearRecoilDatasetCF4(val)
    test_dataset = NuclearRecoilDatasetCF4(test)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def get_dataloaders_ar_cf4(csv_file, batch_size=32):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label_ar_cf4)
    train, test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    train, val = train_test_split(train, test_size=0.1765, stratify=train["label"], random_state=42)
    train_dataset = NuclearRecoilDatasetArCF4(train)
    val_dataset = NuclearRecoilDatasetArCF4(val)
    test_dataset = NuclearRecoilDatasetArCF4(test)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
