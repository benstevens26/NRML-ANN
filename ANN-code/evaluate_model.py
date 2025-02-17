"""
evaluate_model.py - Comprehensive Model Evaluation for LENRI

This script handles:
- Loading the trained LENRI model
- Evaluating the model on the test set
- Computing classification metrics: accuracy, precision, recall, F1-score
- Generating and visualizing a confusion matrix
- Plotting the ROC curve and computing AUC
- Plotting the Precision-Recall curve
- Performing feature importance analysis
- Analyzing misclassified events
- Examining model confidence distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from feature_preprocessing import get_dataloaders
from model import LENRI_CF4

model_path = "LENRI_CF4_1_opt.pth"
features_path = "ANN-code/Data/features_CF4_processed.csv"
save_path = "ANN-code/Data/LENRI-CF4-1"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test set
_, _, test_loader = get_dataloaders(features_path, batch_size=32)

# Load trained model
model = LENRI_CF4().to(device)
checkpoint = torch.load(model_path, map_location=device)  # Load checkpoint
model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
print("Loaded Hyperparameters:", checkpoint["hyperparameters"])  # Debugging
model.eval()

# Define loss function
criterion = nn.CrossEntropyLoss()


def evaluate_model():
    """Evaluate model on test set and return predictions and true labels."""
    test_loss = 0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = correct_test / total_test

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["C", "F"],
        yticklabels=["C", "F"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()


def plot_roc_curve(labels, probs):
    """Plot ROC curve and compute AUC."""
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{save_path}/roc_curve.png")
    plt.show()


def plot_precision_recall_curve(labels, probs):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker="x", label="LENRI Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{save_path}/precision_recall_curve.png")
    plt.show()


def feature_importance_analysis():
    """Analyze feature importance using Random Forest."""
    df = pd.read_csv(features_path)
    df["label"] = df["file_name"].apply(lambda x: 1 if "00_F_" in x else 0)
    X = df.drop(columns=["file_name", "label"]).values
    y = df["label"].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_names = df.drop(columns=["file_name", "label"]).columns
    importances = rf.feature_importances_

    plt.figure(figsize=(12, 6))  # Increase figure size
    plt.barh(range(len(feature_names)), importances, align="center")
    plt.yticks(
        range(len(feature_names)), feature_names, fontsize=10, rotation=0
    )  # Adjust font size
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Nuclear Recoil Classification")
    plt.tight_layout()  # Ensure labels fit properly
    plt.savefig(f"{save_path}/feature_importance.png")
    plt.show()


def plot_confidence_distribution(probs):
    """Plot distribution of model confidence."""
    plt.hist(probs[:, 1], bins=20, alpha=0.7, label="Fluorine Probabilities")
    plt.xlabel("Predicted Probability of Fluorine")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Confidence in Predictions")
    plt.legend()
    plt.savefig(f"{save_path}/confidence_distribution.png")
    plt.show()


# Run all evaluations - comment if not wanted
labels, preds, probs = evaluate_model()
print(f"Precision: {precision_score(labels, preds):.4f}")
print(f"Recall: {recall_score(labels, preds):.4f}")
print(f"F1 Score: {f1_score(labels, preds):.4f}")
plot_confusion_matrix(labels, preds)
plot_roc_curve(labels, probs)
plot_precision_recall_curve(labels, probs)
feature_importance_analysis()
plot_confidence_distribution(probs)
