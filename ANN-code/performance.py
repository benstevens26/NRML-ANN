# single_model_visualizer.py

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import auc, roc_curve
import os

try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    index = script_dir.find("ANN-code")
    if index != -1:  # Ensure the substring exists
        result = script_dir[:index + len("ANN-code")]
        config_path = os.path.join(result, "matplotlibrc.json")
    else:
        print("Something went wrong loading matplotlibrc.json")
        pass
    with open(config_path, "r") as file:  # For reading the matplotlibrc.json file
        custom_params = json.load(file)

    plt.rcParams.update(custom_params)
except:
    print("Something went wrong loading matplotlibrc.json")


def plot_model_performance(
    model_name,
    accuracy=None,
    loss=None,
    val_accuracy=None,
    val_loss=None,
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
    - val_accuracy: List of validation set accuracies over epochs.
    - val_loss: List of validation set loss values over epochs.
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
    if accuracy or loss or val_accuracy or val_loss:
        ax1 = fig.add_subplot(plot_rows, 1, 1)  # Single row for accuracy/loss plot
        lines = []
        if accuracy:
            lines += ax1.plot(accuracy, label="Accuracy", color="blue")
        if val_accuracy:
            lines += ax1.plot(
                val_accuracy, label="Validation Accuracy", color="blue", linestyle="--"
            )
        if loss:
            axloss = ax1.twinx()
            axloss.grid()
            axloss.yaxis.label.set_color("red")
            lines += axloss.plot(loss, label="Loss", color="red")
            axloss.tick_params(axis="y", colors="red")
        if val_loss:
            lines += axloss.plot(
                val_loss, label="Validation Loss", color="red", linestyle="--"
            )
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


def weights_plotter(
    weights,
    names,
    title="Relative Importance of Weights",
    x_label="Input Features",
    y_label="Relative Importance",
):
    """A function to plot the relative importance of model weights.

    Args:
        weights (2D array): 2D array of size (input_ndim, output_ndim)
        containing the weights of each connection in a given layer. Typically
        the result of calling MODEL.layers[n].get_weights()[0].
        names (array): 1D array containing the names of the variables.

    Returns:
        None: doesn't return anything, just plots the relative importance
        of the weights.
    """

    normalised_summed_weights = zscore(
        [np.sum([abs(j[i]) for i in range(weights.shape[1])]) for j in weights]
    )
    colours = ["g" if i > 0 else "r" for i in normalised_summed_weights]
    bars = plt.bar(
        np.arange(0, len(normalised_summed_weights)),
        normalised_summed_weights,
        color=colours,
        edgecolor="black",
        width=0.5,
    )
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1 * (0.75 if bar.get_height() > 0 else -2.5),
            round(bar.get_height(), 2),
            horizontalalignment="center",
        )
    plt.title(title)
    plt.xlabel(x_label)
    plt.xlim(-0.5, len(normalised_summed_weights) + 0.5)
    plt.ylim(min(normalised_summed_weights) - 0.5, max(normalised_summed_weights) + 0.5)
    plt.xticks(np.arange(0, len(normalised_summed_weights)), names, rotation="vertical")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.hlines(
        [0], -0.5, len(normalised_summed_weights) + 0.5, "black", "solid", lw=2.5
    )
    plt.show()
    return None


def roc_plotter(y_true, y_pred_prob):
    """generates a ROC curve using the given inputs and sklearn's implementation.


    Args:
        y_true (numpy array): an array containing the truth values (i.e. an array of 0s and 1s)
        y_pred_prob (numpy array): an array containing the probability of each event being in each output  category

    Returns:
        None: plots the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")  # Dashed diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for LENRI Model")
    plt.legend(loc="lower right")
    plt.grid(visible=True)
    plt.show()

    return None
