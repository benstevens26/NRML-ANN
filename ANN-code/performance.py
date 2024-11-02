# single_model_visualizer.py

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

with open("matplotlibrc.json", "r") as file:
    custom_params = json.load(file)

plt.rcParams.update(custom_params)


def plot_model_performance(
    model_name,
    accuracy=None,
    loss=None,
    confusion_matrix=None,
    precision=None,
    recall=None,
    f1_score=None,
    figsize=(12, 10),
):
    """
    Visualize performance metrics for a single model.

    Parameters:
    - model_name: Name of the model (string).
    - accuracy: List of accuracy values over epochs.
    - loss: List of loss values over epochs.
    - confusion_matrix: 2D array representing the confusion matrix.
    - precision: Precision score.
    - recall: Recall score.
    - f1_score: F1 score.
    - figsize: Tuple specifying the size of the entire figure (default is (12, 8)).
    """

    fig = plt.figure(figsize=figsize)

    # Number of plots
    plot_rows = 2
    plot_cols = 2

    # 1. Accuracy and loss plot (top row spanning both columns)
    if accuracy or loss:
        ax1 = fig.add_subplot(plot_rows, 1, 1)  # Single row for accuracy/loss plot
        lines = []
        if accuracy:
            lines += ax1.plot(accuracy, label="Accuracy", color="blue")
        if loss:
            axloss = ax1.twinx()
            axloss.grid()
            axloss.yaxis.label.set_color("red")
            lines += axloss.plot(loss, label="Loss", color="red", linestyle="--")
            axloss.tick_params(axis="y", colors="red")
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right")
        ax1.set_title(f"{model_name} - Accuracy and Loss over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Value")
        ax1.tick_params(axis="y", colors="blue")
        ax1.grid(True)

    # 2. Confusion Matrix (left side of second row)
    if confusion_matrix is not None:
        ax2 = fig.add_subplot(plot_rows, plot_cols, 3)  # Left plot in second row
        sns.heatmap(
            confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2
        )
        ax2.set_title(f"{model_name} - Confusion Matrix")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

    # 3. Precision, Recall, F1-score Table (right side of second row)
    if precision is not None or recall is not None or f1_score is not None:
        # Create DataFrame for table display
        table_data = {
            "Metric": ["Precision", "Recall", "F1 Score"],
            "Score": [
                f"{precision:.2f}" if precision is not None else "N/A",
                f"{recall:.2f}" if recall is not None else "N/A",
                f"{f1_score:.2f}" if f1_score is not None else "N/A",
            ],
        }
        df_metrics = pd.DataFrame(table_data)

        ax3 = fig.add_subplot(plot_rows, plot_cols, 4)  # Right plot in second row
        ax3.axis("tight")
        ax3.axis("off")
        table = ax3.table(
            cellText=df_metrics.values,
            colLabels=df_metrics.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(df_metrics.columns))))
        table.scale(1, 4)
        table.set_fontsize(18)
        ax3.set_title(f"{model_name} - Precision, Recall, and F1 Score")

    fig.tight_layout(pad=3.0)
    plt.show()
