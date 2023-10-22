import numpy as np
from LinRegression import LinRegression
from GD_and_SGD_analytical import GD, SGD
from GD_and_SGD_AD import AD_GD, AD_SGD

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


## Want to try GD setup without momentum

beta_GD = GD(linreg.X, linreg.y, initial_step, n_epochs, tol, regression_method='OLS')



## Want to try SGD setup with Ridge and "Normal momentum"
beta = SGD(linreg.X, linreg.y, batch_size, n_epochs, initial_step, momentum, tol, regression_method='OLS')



# trying AD with GD momentum ols

beta_AD_mom = AD_GD(linreg.X, linreg.y, initial_step, max_iter, tol, regression_method='OLS', momentum=momentum, optimization='momentum')
beta_AD_no_mom = AD_GD(linreg.X, linreg.y, initial_step, max_iter, tol, regression_method='OLS')

print(beta_AD_mom)

# SGD ols momentum adam  ( rho1 og rho vanligvis mellom 0.9 og 0.99? ) # denne tar ekstremt mye lenger tid, og får annerledes resultat.
beta_adam_AD_SGD = AD_SGD(linreg.X, linreg.y, batch_size, n_epochs, initial_step, momentum, tol, regression_method='OLS', optimization='adam', rho1 = 0.9, rho2 = 0.99)
print(beta_adam_AD_SGD)


print(beta_ols)