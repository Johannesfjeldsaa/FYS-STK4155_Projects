# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:21:45 2023

@author: vildesn
"""

from Franke_function import FrankeFunction
from LinRegression import LinRegression
from sklearn import linear_model
from Plotting import Plotting
from setup import save_fig, data_path

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')

np.random.seed(2500)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

z = FrankeFunction(x, y)

MSE_test_scores = []
R2_test_scores = []
beta_parameters = []
polydegree = 1

OLS_regression = LinRegression(polydegree, x, y, z) #create class
OLS_regression.split_data(1/5) # perform split of data

OLS_regression.scale(scaling_method='StandardScaling')

la = 0.0001
# Training the data
RegLasso = linear_model.Lasso(la, fit_intercept=False)
RegLasso.fit(OLS_regression.X_train_scaled[:, 1:], OLS_regression.y_train_scaled)

beta = RegLasso.coef_

ypredictLasso = RegLasso.predict(OLS_regression.X_test_scaled[:, 1:])
y_pred_test = (OLS_regression.X_test_scaled[:, 1:] @ beta) + OLS_regression.y_scaler
        
MSE_test = OLS_regression.MSE(OLS_regression.y_test, y_pred_test)
        
        
        
        
        
        
        
        
        
        
        
        