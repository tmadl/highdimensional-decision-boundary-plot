from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def dshape(X):
    if len(X.shape) == 1:
        return X.reshape(-1, 1)
    else:
        return X if X.shape[0] > X.shape[1] else X.T


def unpack(t):
    while type(t) == list or type(t) == np.ndarray:
        t = t[0]
    return t


def tonumeric(lst):
    lbls = {}
    for t in lst.flatten():
        if unpack(t) not in lbls:
            lbls[unpack(t)] = len(lbls.keys())
    return np.array([lbls[unpack(t)] for t in lst.flatten()])


def getdataset(datasetname, onehot_encode_strings=True):
    # load
    dataset = fetch_openml(datasetname)
    # get X and y
    X = dshape(dataset.data)
    try:
        target = dshape(dataset.target)
    except:
        print("WARNING: No target found. Taking last column of data matrix as target")
        target = X[:, -1]
        X = X[:, :-1]
    if (
        len(target.shape) > 1 and target.shape[1] > X.shape[1]
    ):  # some mldata sets are mixed up...
        X = target
        target = dshape(dataset.data)
    if len(X.shape) == 1 or X.shape[1] <= 1:
        for k in dataset.keys():
            if k != "data" and k != "target" and len(dataset[k]) == X.shape[1]:
                X = np.hstack((X, dshape(dataset[k])))
    # one-hot for categorical values
    if onehot_encode_strings:
        cat_ft = [
            i
            for i in range(X.shape[1])
            if "str" in str(type(unpack(X[0, i])))
            or "unicode" in str(type(unpack(X[0, i])))
        ]
        if len(cat_ft):
            for i in cat_ft:
                X[:, i] = tonumeric(X[:, i])
            X = OneHotEncoder(categorical_features=cat_ft).fit_transform(X)
    # if sparse, make dense
    try:
        X = X.toarray()
    except:
        pass
    # convert y to monotonically increasing ints
    y = tonumeric(target).astype(int)
    return np.nan_to_num(X.astype(float)), y
