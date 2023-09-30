# testdata_tcp.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
from numpy import dot
from numpy.linalg import norm
import os
import joblib

def preprocess_and_generate_test_data_tcp():
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.6f}'.format})

    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, "Dataset/dataset.csv")
    dataset = pd.read_csv(dataset_path)
    dataset.fillna(0, inplace=True)

    data = dataset.values
    X = data[:, 0:data.shape[1] - 1]
    Y = data[:, data.shape[1] - 1]

    pca = PCA(n_components=10, whiten=False)
    X = pca.fit_transform(X)

    rf = RandomForestRegressor()
    rf.fit(X, Y)

    rf_model_path = os.path.join(script_dir, "model", "rf_model.pkl")
    joblib.dump(rf, rf_model_path)

    if os.path.exists(os.path.join(script_dir, "model/kde.pckl")):
        with open(os.path.join(script_dir, "model/kde.pckl"), 'rb') as f:
            kde = pickle.load(f)
    else:
        kde = KernelDensity(kernel='gaussian')
        kde.fit(X)
        with open(os.path.join(script_dir, "model/kde.pckl"), 'wb') as f:
            pickle.dump(kde, f)

    run = True
    y_label = []
    predict = None
    mse = None
    while run:
        y_label.clear()
        new_test_data = np.abs(kde.sample(50))
        for i in range(len(new_test_data)):
            label = getLabel(X, Y, new_test_data[i])
            y_label.append(label)
        y_pred = np.asarray(y_label)
        predict = rf.predict(new_test_data)
        mse = mean_squared_error(y_pred, predict)
        if mse < 0.80:
            run = False
    y_label = np.asarray(y_label)

    results = []
    for i in range(len(y_label)):
        result = {
            "Generated Test Data": new_test_data[i].tolist(),
            "Original Network Delay": float(y_label[i]),
            "Predicted Delay": float(predict[i])
        }
        results.append(result)

    return results, mse

def getLabel(train, label, test_data):
    output = 0
    similarity = 0
    for i in range(len(train)):
        sim = dot(train[i], test_data) / (norm(train[i]) * norm(test_data))
        if sim > similarity:
            output = label[i]
            similarity = sim
    return output
