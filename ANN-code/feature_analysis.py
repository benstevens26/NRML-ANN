import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

with open("matplotlibrc.json", "r") as file:
    custom_params = json.load(file)

plt.rcParams.update(custom_params)
model_name = "LENRI"
def feature_analysis():
    df = pd.read_csv("Data/more_features_noisy.csv")

    # choose actual features (so ignore name and dark frame columns)
    features = df.drop(df.columns[:2], axis=1)

    carbon = df[df["name"].str.contains("C")]
    fluorine = df[df["name"].str.contains("F")]

    for feature in features:
        plt.figure(figsize=(10, 6))

        sns.histplot(
            carbon[feature],
            label="Carbon",
            color="blue",
            alpha=0.5,
            bins=50,
            stat="density",
        )
        sns.histplot(
            fluorine[feature],
            label="Fluorine",
            color="orange",
            alpha=0.5,
            bins=50,
            stat="density",
        )

        plt.title("Comparison of " + feature + ": Carbon vs Fluorine")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("figures/feature-analysis/" + model_name + "/" + feature + ".png")
        plt.show()


feature_analysis()
