"""
train.py - Training script for LENRI classification

This script handles:
- Loading the dataset using feature_preprocessing.py
- Defining the model, loss function, and optimizer
- Training the model with mini-batch gradient descent
- Logging training and validation loss for monitoring
- Implementing early stopping to prevent overfitting
- Saving the trained model
"""

import torch
import torch.optim as optim
import torch.nn as nn
from feature_preprocessing import get_dataloaders
from model import LENRI

# Hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 32
patience = 5  # Early stopping patience

# Load datasets
train_loader, val_loader, _ = get_dataloaders("data/features_raw.csv", batch_size=batch_size)

# Initialize model, loss function, optimizer
model = LENRI()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
best_val_loss = float("inf")
stopping_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {total_loss:.4f} - Validation Loss: {val_loss:.4f}")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stopping_counter = 0
        torch.save(model.state_dict(), "lenri_model.pth")  # Save best model
    else:
        stopping_counter += 1
        if stopping_counter >= patience:
            print("Early stopping triggered. Training halted.")
            break

print("Training complete. Best model saved as lenri_model.pth")
