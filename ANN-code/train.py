"""
train.py - Training script for LENRI

This script handles:
- Loading the dataset using feature_preprocessing.py
- Defining the model, loss function, and optimizer
- Training the model with mini-batch gradient descent
- Logging training and validation loss and accuracy
- Implementing early stopping to prevent overfitting
- Saving the best-performing model
"""

import torch
import torch.optim as optim
import torch.nn as nn
from feature_preprocessing import get_dataloaders
from model import LENRI_CF4

# Hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 32
patience = 5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
train_loader, val_loader, _ = get_dataloaders("ANN-code/Data/features_CF4_processed.csv", batch_size=batch_size)

# Initialize model, loss function, optimizer
model = LENRI().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
best_val_acc = 0.0
stopping_counter = 0

# exit() # remove to train the model
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)
    
    train_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)
    
    val_loss /= len(val_loader)
    val_acc = correct_val / total_val
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        stopping_counter = 0
        torch.save(model.state_dict(), "lenri_model_best.pth")  # Save best model
        print("Best model updated.")
    else:
        stopping_counter += 1
        if stopping_counter >= patience:
            print("Early stopping triggered. Training halted.")
            break

print("Training complete. Best model saved as lenri_model_best.pth")