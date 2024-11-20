from LENRI import *
from scipy.optimize import minimize

true_C = []
false_C = []
true_F = []
false_F = []

for i in range(len(X_test)):
    cut = X_test[i : i + 1]
    pred = LENRI.predict(cut)
    if np.argmax(pred) == 0:
        if y_true[i] == 0:
            true_C.append(cut)
        elif y_true[i] == 1:
            false_C.append(cut)
        else:
            print("ERROR")
    elif np.argmax(pred) == 1:
        if y_true[i] == 0:
            false_F.append(cut)
        elif y_true[i] == 1:
            true_F.append(cut)
        else:
            print("ERROR")


true_C = np.vstack([arr.flatten() for arr in true_C])
false_C = np.vstack([arr.flatten() for arr in false_C])
true_F = np.vstack([arr.flatten() for arr in true_F])
false_F = np.vstack([arr.flatten() for arr in false_F])


true_C_probs = LENRI.predict(true_C)[:, 1]
false_C_probs = LENRI.predict(false_C)[:, 1]
true_F_probs = LENRI.predict(true_F)[:, 1]
false_F_probs = LENRI.predict(false_F)[:, 1]

plt.plot(true_F_probs, "x", color="green")
plt.plot(false_C_probs, "x", color="red")
plt.plot(
    np.linspace(0, len(true_F), len(true_C)), true_C_probs, "o", ms=4, color="green"
)
plt.plot(
    np.linspace(0, len(true_F), len(false_F)), false_F_probs, "o", ms=4, color="red"
)
plt.plot([-1, -1], [-1, -1], "o", ms=4, color="gray", label="Carbon")
plt.plot([-1, -1], [-1, -1], "x", color="gray", label="Fluorine")
plt.plot([-1, -1], [-1, -1], "s", color="green", label="Correct prediction")
plt.plot([-1, -1], [-1, -1], "s", color="red", label="Incorrect prediction")
plt.plot(
    [0, len(true_F)],
    [0.5865865865865866, 0.5865865865865866],
    "--",
    color="black",
    label="Threshold = 0.587",
)
plt.xlim(-50, len(true_F) + 50)
plt.ylabel("Prediction")
plt.ylim(-0.02, 1.02)
plt.xlabel("Event # (loosely energy-related)")
plt.plot()
plt.legend(loc="lower right", framealpha=0.9)
plt.text(-90, 1.03, "Fluorine", horizontalalignment="right", verticalalignment="bottom")
plt.text(-90, -0.07, "Carbon", horizontalalignment="right", verticalalignment="bottom")
plt.title("Predicted Probabilities")
plt.show()


thresholds = np.linspace(0.5, 1, 500)
wrong_C_fraction = []
right_F_fraction = []
for threshold in thresholds:
    wrong_C_fraction.append(
        len([i for i in false_F_probs if i < threshold]) / len(false_F_probs)
    )
    right_F_fraction.append(
        len([i for i in true_F_probs if i > threshold]) / len(true_F_probs)
    )

plt.plot(thresholds, wrong_C_fraction, color="blue", label="C")
plt.plot(thresholds, right_F_fraction, color="orange", label="F")
plt.xlabel("Threshold")
plt.ylabel("Fraction")
plt.legend()
plt.show()


def threshold_function(threshold, y_pred_prob, y_true):
    t_pred = [1 if pred_prob > threshold else 0 for pred_prob in y_pred_prob]
    threshold_f1 = f1_score(y_true, t_pred, average="weighted")
    return 1 - threshold_f1


# result = minimize(
#     threshold_function,
#     x0=0.58,
#     args=(y_pred_prob, y_true),
#     bounds=[(0, 1)],
#     method='L-BFGS-B'  # Specify a bounded optimization method
# )
# best_thresh = result.x[0]
# # ^^This shit didn't work, so I'm just going to use the "find the maximum threshold out of 1000 tests" approach.
