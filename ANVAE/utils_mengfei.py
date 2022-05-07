# -*- coding: utf-8 -*-
"""
@File    : utils_mengfei.py
@Time    : 8/5/2021 3:44 PM
@Author  : Mengfei Yuan
"""

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def plot_zmeans(z_means, y_test, filename):
    plt.figure(figsize=(12, 10))
    plt.scatter(z_means[:, 0], z_means[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    # plt.show()

def get_classification_report(y_true, y_pred):
    """y_pred is the output after sigmoid/softmax"""
    y_pred_01 = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))], dtype='uint8')

    report = classification_report(y_true, y_pred_01)
    acc = accuracy_score(y_true, y_pred_01)

    print('classification_report:\n', report)
    return y_pred_01, acc, report

def plot_true_prediction(y_pred, y_true, filename):
    c_pred = [i[0] for i in y_pred]
    k_pred = [i[1] for i in y_pred]
    c_true = [i[0] for i in y_true]
    k_true = [i[1] for i in y_true]

    plt.figure(figsize=(12,10))
    plt.plot(c_true, c_true, 'b-')
    plt.plot(c_true, c_pred, 'r*')
    plt.xlabel("true initial composition")
    plt.ylabel("predicted initial composition")
    plt.savefig(filename + '.init_comp.png')

    plt.figure(figsize=(12,10))
    plt.plot(k_true, k_true, 'b-')
    plt.plot(k_true, k_pred, 'r*')
    plt.xlabel("true interfacial energy")
    plt.ylabel("predicted interfacial energy")
    plt.savefig(filename + '.interf_energy.png')

    with open(filename + 'true_pred_data.txt', 'w') as f:
        for i in range(len(c_pred)):
            f.write(str(c_true[i])+'\t'+str(c_pred[i])+'\t'+str(k_true[i])+'\t'+str(k_pred[i])+'\n')
