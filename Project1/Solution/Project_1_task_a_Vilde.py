# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 07:57:54 2023

@author: vildesn
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def make_design_matrix(x, y, degree):
    
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    
    N = len(x)
    l = int((degree+1)*(degree+2)/2)
    X = np.ones((N, l))
    
    for i in range(1, degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
            
    return X

def scale_with_mean(X_train, X_test, y_train):
    "Scales the data by subtracting the mean"
    
    X_train_mean = np.mean(X_train[:, 1:], axis=0)
    
    X_train_scaled = X_train[:, 1:] - X_train_mean # Removes intercept
 
    X_test_scaled = X_test[:, 1:] - X_train_mean

    y_train_scaled = y_train - np.mean(y_train)
    
    return X_train_scaled, X_test_scaled, y_train_scaled

def find_beta_OLS(X, y):
    "Finding the betas using analytical solution of ordinary least squares"
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Make data.
#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)
#x, y = np.meshgrid(x,y)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))


z = FrankeFunction(x, y)

degrees = [2]

for degree in degrees:
    print(f"Degree {degree}")
    X = make_design_matrix(x, y, degree)

    x_train, x_test, y_train, y_test = train_test_split(X, np.ravel(z), test_size=0.2)
    
    # Scaling the data:
    x_train_scaled, x_test_scaled, y_train_scaled = scale_with_mean(x_train, x_test, y_train)
    
    beta_OLS = find_beta_OLS(x_train_scaled, y_train_scaled)

    print(beta_OLS)

    y_tilde_OLS = x_train @ beta_OLS 
    
    y_pred_OLS = x_test @ beta_OLS + np.mean(y_train)  
    
    print("MSE and R2 train: ")
    print(mean_squared_error(y_train, y_tilde_OLS))
    print(r2_score(y_train, y_tilde_OLS))

    print("MSE and R2 test: ")
    print(mean_squared_error(y_test, y_pred_OLS))
    print(r2_score(y_test, y_pred_OLS))


#MSE_train.loc[degree, "OLS"] = mean_squared_error(y_train_scaled, y_tilde_OLS)
#MSE_test.loc[degree, "OLS"] = mean_squared_error(y_test, y_pred_OLS)
#%%

# Plot the surface.
fig = plt.figure()
ax = plt.axes(projection="3d")
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()