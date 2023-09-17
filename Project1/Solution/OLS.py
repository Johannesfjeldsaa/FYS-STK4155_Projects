### Import packages ###

import os

import pandas as pd
import regex as re

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')

from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error


### Where to save the figures and data files ###

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')


### 1D OLS ###

def preprocess(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)  # Deler opp modeldataen i trenings og test-data

    # Model training, we compute the mean value of y and X
    y_train_mean = np.mean(y_train)
    X_train_mean = np.mean(X_train, axis=0)
    X_train = X_train - X_train_mean
    y_train = y_train - y_train_mean

    # Model prediction, we need also to transform our data set used for the prediction.
    X_test = X_test - X_train_mean  # Use mean from training data

    return X_train, X_test, y_train, y_test, X_train_mean, y_train_mean


def OLS_and_cost_calc(X, y):
    X_train, X_test, y_train, y_test, X_train_mean, y_train_mean = preprocess(X, y, test_size=.4)

    clf = skl.LinearRegression().fit(X_train, y_train)
    fity = clf.predict(X)
    linreg = LinearRegression()
    beta = linreg.fit(X_train, y_train)

    y_tilde = (X_train @ beta)  # y_train og x_train er skalert, siden y_tilde kun brukes videre til MSE_train med y_train trenger ikke denne skalering
    y_predicted = linreg.predict(X_test) + y_train_mean  # y_predicted skal brukes til MSE_test med y_test som IKKE er skalert, vi må derfor skalere y_predicted tilbake.

    MSE_train = mean_squared_error(y_train, y_tilde)
    MSE_test = mean_squared_error(y_test, y_predicted)

    return MSE_train, MSE_test