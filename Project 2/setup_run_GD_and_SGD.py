import numpy as np
from LinRegression import LinRegression
from GD_and_SGD_analytical import GD, SGD
from GD_and_SGD_AD import AD_GD, AD_SGD
import matplotlib.pyplot as plt

## set up data

np.random.seed(1997)
n = 1000
x = 2*np.random.rand(n, 1)
y = 4 + 3*x + 2*x*x + np.random.randn(n,1)
linreg = LinRegression(2, x, y)

n_epochs = 50
batch_size = 10
max_iter = 500
lambd = 0.1
tol = 10**-5
momentum = 0.3 # momentum parameter
step_size = 0.1
initial_step = 0.1

# used for ridge
XT_X = linreg.X.T @ linreg.X
Id = n*lambd* np.eye(XT_X.shape[0])  #identitymatrix

# For performing educated guess of step size
# Hessian matrix
H_ridge = (2.0/n)* XT_X+2*lambd* np.eye(XT_X.shape[0])
H_ols = (2.0/n) * linreg.X.T @ linreg.X
# Get the eigenvalues
EigValues_ols, _ = np.linalg.eig(H_ols)
EigValues_ridge, _ = np.linalg.eig(H_ridge)

#orig beta values for comparison
XT_X = linreg.X.T @ linreg.X
Id = n*lambd* np.eye(XT_X.shape[0])  #identitetsmatrise
beta_ridge = np.linalg.inv(XT_X+Id) @ linreg.X.T @ y
beta_ols = np.linalg.inv(linreg.X.T @ linreg.X) @ linreg.X.T @ y


# for run OLS, analytical, SGD, all methods . collect betas and MSE_scores
optimization_methods = ['no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam']
analysis_dict = {'no momentum': [70, 700, 100, 300], 'momentum': [5, 200, 5, 300], 'RMSprop': [40, 700, 20, 100], 'adagrad': [40, 500, 100, 100], 'adam': [70, 1000, 70, 1000]}



for key, value in analysis_dict.items():
    beta, MSE = SGD(linreg.X, linreg.y, value[0], value[1], 0.1, 0.3,
                    10**-8, regression_method='OLS', optimization=key)
    y_pred = linreg.X@beta



    #plt.plot(sorted(x), sorted(y_pred), label=key)

    plt.plot(range(len(MSE))[::1000], MSE[::1000], label=key)  # prøvde på en konvergensgraf

#plt.plot(x[::15],y[::15],'ro')
plt.title('Convergencegraph: analytical SGD analysis for OLS')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.show()

