# -*- coding: utf-8 -*-
"""
@File    : label2params.py
@Time    : 8/3/2021 3:39 PM
@Author  : Mengfei Yuan
"""
import numpy as np

# map spinodal parameters to class
label2param = {
    1:[0, 0.25],
    2:[0, 0.7],
    3:[0.1, 0.25],
    4:[0.1, 0.7],
    5:[0.2, 0.25],
    6:[0.2, 0.7],
}

def get_params(label, label2params):
    """ covert label to real parameters"""
    params = []
    label = label.squeeze()
    for i in label:
        params.append(label2params[i])
    return np.array(params)


if __name__ == "__main__":
    true = list(label2param.values())
    print(true)
    ic, kappa  = [], []
    for i in true:
        ic.append(i[0])
        kappa.append(i[1])

    print(ic)
    print(kappa)

    predict = [[0.06, 0.249], [0.032, 0.71], [0.13, 0.26], [0.09, 0.679], [0.18, 0.23], [0.205, 0.704]]
    ic_p, kappa_p  = [], []
    for i in predict:
        ic_p.append(i[0])
        kappa_p.append(i[1])

    print(ic_p)
    print(kappa_p)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    plt.plot(ic, ic, '-o')
    plt.plot(ic, ic_p, 'r*')
    plt.xlabel('True initial condition')
    plt.ylabel('Predicted initial condition')
    plt.savefig('ic.png')

    plt.figure(figsize=(5,3))
    plt.plot(kappa, kappa, '-o')
    plt.plot(kappa, kappa_p, 'r*')
    plt.xlabel('True interfacial energy')
    plt.ylabel('Predicted interfacial energy')
    plt.savefig('kappa.png')