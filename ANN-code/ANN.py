import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from analysis import *
from convert_sim_ims import *
from feature_extraction import *
from sklearn.neural_network import MLPClassifier

X = [[0.0, 0.0], [1.0, 1.0]]
y = [0, 1]
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)
