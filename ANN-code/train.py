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
from feature_preprocessing import get_dataloaders_cf4
from feature_preprocessing import get_dataloaders_ar_cf4
from model import LENRI_CF4
from model import LENRI_Ar_CF4

# model to train and features to use
model = LENRI_CF4(input_size=10)
features_path = "ANN-code/Data/features_old/all_features_2_scaled.csv"
binary = True

# Hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 32
patience = 5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
if binary:
    train_loader, val_loader, _ = get_dataloaders_cf4(features_path, batch_size=batch_size)
else:
    train_loader, val_loader, _ = get_dataloaders_ar_cf4(features_path, batch_size=batch_size)

# Initialize model, loss function, optimizer
model = LENRI_CF4(input_size=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
best_val_acc = 0.0
stopping_counter = 0

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

        # Prepare checkpoint dictionary
        checkpoint = {
            "model_state_dict": model.state_dict(),  # Save model weights
            "optimizer_state_dict": optimizer.state_dict(),  # Save optimizer state (optional)
            "best_val_acc": best_val_acc,  # Store best validation accuracy
        }

        # Save hyperparameters if they exist, else save "None"
        if "hyperparameters" not in locals():
            hyperparameters = None
            checkpoint["hyperparameters"] = hyperparameters
        else:
            checkpoint["hyperparameters"] = hyperparameters
        
        # Save model checkpoint
        torch.save(checkpoint, "lenri_model_best.pth")
        
        print("Best model updated.")
    else:
        stopping_counter += 1
        if stopping_counter >= patience:
            print("Early stopping triggered. Training halted.")
            break
        

print("Training complete. Best model saved as lenri_model_best.pth")