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