"""
hyperparam_tuning.py - Hyperparameter and Architecture Tuning for LENRI

This script handles:
- Defining the LENRI model with dynamic architecture
- Implementing hyperparameter tuning using Optuna
- Training LENRI with various hyperparameters and architectures
- Selecting the best performing configuration
- Saving the best model and hyperparameters
"""

import torch
import torch.optim as optim
import torch.nn as nn
import optuna
from feature_preprocessing import get_dataloaders_cf4

features_path = "ANN-code/Data/features_old/all_features_2_scaled.csv"


def save_best_model(model, trial):
    """Saves the best model with its hyperparameters."""
    model_path = f"lenri_best_{trial.number}.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "hyperparameters": trial.params},
        model_path,
    )
    print(f"Best model saved as {model_path}")


class LENRI(nn.Module):
    def __init__(
        self,
        input_size=9,
        hidden_layers=[32, 16, 8],
        activation="LeakyReLU",
        dropout_rate=0.2,
    ):
        super(LENRI, self).__init__()

        layers = []
        prev_size = input_size

        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, layer_size))

            if activation == "ReLU":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.01))

            layers.append(nn.Dropout(dropout_rate))
            prev_size = layer_size

        layers.append(nn.Linear(prev_size, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(
    learning_rate,
    batch_size,
    dropout_rate,
    optimizer_name,
    hidden_layers,
    activation,
    trial,
    num_epochs=50,
    patience=5,
):
    """Trains LENRI with given hyperparameters and architecture and returns validation accuracy."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders_cf4(
        features_path, batch_size=batch_size
    )

    model = LENRI(
        hidden_layers=hidden_layers, activation=activation, dropout_rate=dropout_rate
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = (
        optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer_name == "Adam"
        else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    )

    best_val_acc = 0.0
    stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
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

            predictions = torch.argmax(outputs, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train

        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stopping_counter = 0
            save_best_model(model, trial)
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered. Training halted.")
                break

    return best_val_acc


def objective(trial):
    """Objective function for Optuna hyperparameter tuning with refined search space."""

    learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    activation = trial.suggest_categorical("activation", ["LeakyReLU", "ReLU"])

    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_layers = [
        trial.suggest_int(f"n_units_layer_{i}", 50, 100) for i in range(num_layers)
    ]

    val_acc = train_model(
        learning_rate,
        batch_size,
        dropout_rate,
        optimizer_name,
        hidden_layers,
        activation,
        trial,
    )
    return val_acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best hyperparameters:", study.best_params)
