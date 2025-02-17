"""
test_model.py - Model Evaluation Script for LENRI

This script handles:
- Loading the trained LENRI model
- Evaluating the model on the test set
- Computing test loss and accuracy
- Calculating precision, recall, and F1-score
- Providing insights into model performance
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from feature_preprocessing import get_dataloaders_cf4, get_dataloaders_ar_cf4
from model import LENRI_CF4, LENRI_Ar_CF4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "LENRI_CF4_old_unopt.pth"
features_path = "ANN-code/Data/features_old/all_features_2_scaled.csv"
model = LENRI_CF4().to(device)
binary = True

# Load test set
if binary:
    _, _, test_loader = get_dataloaders_cf4(features_path, batch_size=32)
else:
    _, _, test_loader = get_dataloaders_ar_cf4(features_path, batch_size=32)

# Load trained model
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Define loss function
criterion = nn.CrossEntropyLoss()

# Evaluate on test set
test_loss = 0
correct_test = 0
total_test = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_test += (predictions == labels).sum().item()
        total_test += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())

# Compute test loss and accuracy
test_loss /= len(test_loader)
test_acc = correct_test / total_test

# Compute additional metrics
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

# Print results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
