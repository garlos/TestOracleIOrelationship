# Print test & predicted & difference data in terminal
import numpy as np


def print_values(y_test, pred):
    for i in range(len(y_test)):
        y_test[i] = np.around(y_test[i] * 2755, 1)
        pred[i] = np.around(pred[i] * 2755, 1)
        diff = np.around(y_test[i] - pred[i], 1)
        print("Test=%s, Pred=%s, Diff=%s" % (y_test[i], pred[i], diff))
