import numpy as np
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
from LinRegression import LinRegression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import random

" GD for OLS and Ridge"

def gd_plain(X, y, step_size, max_iter, lambd=None, regression_method=None):

    MSE_gd_ols = []
    MSE_gd_ridge = []
    p = X.shape[1]
    beta = np.random.randn(p, 1)

    for iter in range(max_iter):

        if regression_method == 'OLS':
            grad_ols = (2.0/n) * X.T @ (X @ beta - y)  # analytical expression of gradient
            beta -= step_size*grad_ols
            MSE_gd_ols.append(np.mean((y - X @ beta) ** 2))

            if np.linalg.norm(grad_ols) < tol:  # check convergence criterion
                print("Converged!")
                break

        elif regression_method == 'Ridge':
            grad_ridge = 2.0 / n * X.T @ (X @ (beta) - y) + 2*lambd * beta
            beta -= step_size * grad_ridge
            MSE_gd_ridge.append(np.mean((y - X @ beta) ** 2))

            if np.linalg.norm(grad_ridge) < tol:  # check convergence criterion
                print("Converged!")
                break
        else:
            raise ValueError ('Valid regression method was not given')

    return beta, MSE_gd_ols, MSE_gd_ridge

"Under is GD with momentum for OLS and Ridge"
def ols_objective(X, y, beta):
  # X is the feature matrix (n x p)
  # y is the response vector (n x 1)
  # w is the coefficient vector (p x 1)
  n = X.shape[0] # number of samples
  y_pred = X @ beta # predicted response (n x 1)
  error = y - y_pred # error vector (n x 1)
  sse = np.sum(error**2) / n # sum of squared errors
  obj = sse # objective function value
  return obj

def ridge_objective(X, y, beta, lambd):
  # X is the feature matrix (n x p)
  # y is the response vector (n x 1)
  # beta is the coefficient vector (p x 1)
  # lambda_ is the regularization parameter
  n = X.shape[0]
  y_pred = X @ beta # predicted response (n x 1)
  sse = (np.sum((y-y_pred)**2)) / n # sum of squared errors
  reg = lambd * np.sum(beta**2) # regularization term
  obj = sse + reg # objective function value
  return obj

def ols_gradient(X, y, beta):
  # X is the feature matrix (n x p)
  # y is the response vector (n x 1)
  # beta is the coefficient vector (p x 1)

  n = X.shape[0] # number of samples
  grad = (2.0/n) * X.T @ (X @ beta - y) # gradient vector (p x 1)
  return grad

def ridge_gradient(X, y, beta, lambd):
  # X is the feature matrix (n x p)
  # y is the response vector (n x 1)
  # beta is the coefficient vector (p x 1)
  # lambd is the regularization parameter
  n = X.shape[0] # number of samples
  y_pred = X @ beta # predicted response (n x 1)
  error = y - y_pred # error vector (n x 1)
  grad = -2/n * X.T @ error + 2 * lambd * beta # gradient vector (p x 1)
  return grad


#gradients = 2.0 / n * linreg.X.T @ (linreg.X @ (beta_ridge) - y) + 2 * lmbda * beta_ridge


def gd_momentum(X, y, step_size, momentum, max_iter, tol, lambd=None, regression_method=None):
    # X is the feature matrix (n x p)
    # y is the response vector (n x 1)
    # step_size is the learning rate
    # beta is the momentum parameter
    # lambda_ is the regularization parameter
    # max_iter is the maximum number of iterations
    # tol is the tolerance for convergence criterion


    p = X.shape[1]  # number of features
    opt_beta = np.random.randn(p, 1)  # initialize coefficient vector (p x 1) # for the opt beta? start guess = 0
    change_vector = np.zeros((p, 1))  # initialize velocity vector (p x 1), change vector, to be updated
    MSE_ols_GD_momentum = []
    MSE_ridge_GD_momentum = []

    for i in range(max_iter):

        if regression_method == 'OLS':
            obj = ols_objective(X, y, opt_beta)  # compute objective function value
            MSE_ols_GD_momentum.append(obj)
            grad = ols_gradient(X, y, opt_beta)  # compute gradient vector (p x 1)
            change_vector = momentum * change_vector - step_size * grad  # update velocity vector (p x 1) with momentum
            opt_beta = opt_beta + change_vector  # update coefficient vector (p x 1) with velocity

            if np.linalg.norm(grad) < tol:  # check convergence criterion
                print("Converged!")
                break


        elif regression_method == 'Ridge':
            obj = ridge_objective(X, y, opt_beta, lambd)  # compute objective function value
            MSE_ridge_GD_momentum.append(obj)
            grad = ridge_gradient(X, y, opt_beta, lambd)  # compute gradient vector (p x 1)
            change_vector = momentum * change_vector - step_size * grad  # update velocity vector (p x 1) with momentum
            opt_beta = opt_beta + change_vector  # update coefficient vector (p x 1) with velocity

            if np.linalg.norm(grad) < tol:  # check convergence criterion
                print("Converged!")
                break

        else:
            print('Regression method was not selected!')


    return opt_beta, MSE_ols_GD_momentum, MSE_ridge_GD_momentum

#set up
np.random.seed(1997)
n = 100
x = 2*np.random.rand(n, 1)
y = 4 + 3*x + 2*x*x + np.random.randn(n,1)
linreg = LinRegression(2, x, y)

max_iter = 50
lambd = 0.01
tol = 10**-5
momentum = 0.3 # momentum parameter
step_size = 0.1

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

eta_guess_ridge = 1.0/np.max(EigValues_ridge)  #educated guess
eta_guess_ols = 1.0/np.max(EigValues_ols)  #educated guess

beta_gd_ols, MSE_gd_ols, _ = gd_plain(linreg.X, linreg.y, step_size, max_iter, regression_method='OLS')
beta_gd_ridge, _, MSE_gd_ridge = gd_plain(linreg.X, linreg.y, step_size, max_iter, lambd=lambd, regression_method='Ridge')
beta_gd_mom_ridge, _, MSE_gdmom_ridge = gd_momentum(linreg.X, linreg.y, step_size, momentum, max_iter, tol, lambd=lambd, regression_method='Ridge')
beta_gd_mom_ols, MSE_gdmom_ols, _ = gd_momentum(linreg.X, linreg.y, step_size, momentum, max_iter, tol, regression_method='OLS')

"""
Compare beta-param result and plot difference
"""
XT_X = linreg.X.T @ linreg.X
Id = n*lambd* np.eye(XT_X.shape[0])  #identitetsmatrise
beta_ridge = np.linalg.inv(XT_X+Id) @ linreg.X.T @ y
beta_ols = np.linalg.inv(linreg.X.T @ linreg.X) @ linreg.X.T @ y

#scikit
sgdreg = SGDRegressor(max_iter = max_iter, eta0=step_size, tol=tol, fit_intercept=False) # nå tas intercept med! "antatt at data er skalert"
sgdreg.fit(linreg.X, linreg.y.ravel())
intercept_scikit = sgdreg.intercept_
weights_scikit = np.array(sgdreg.coef_)
number_iterations = sgdreg.n_iter_
print('number iterations scikit:')
print(number_iterations)

y_predict_scikit = linreg.X @ weights_scikit
y_predict_ols = linreg.X @ beta_gd_mom_ols
y_predict_ridge = linreg.X @ beta_gd_mom_ridge
y_predict_regression_ols = linreg.X @ beta_ols
y_predict_regression_ridge = linreg.X @ beta_ridge

# eventuelt ta med scikit også..
plt.plot(x,y,'ro', label='actual data')
plt.plot(sorted(x), sorted(y_predict_ols.ravel()), 'b-', label='gradient: ols')
plt.plot(sorted(x), sorted(y_predict_ridge.ravel()), 'm-', label='gradient: ridge')# Blir det feil å sortere dem???
plt.plot(sorted(x), sorted(y_predict_regression_ols.ravel()), '-c', label='regression: ols')
plt.plot(sorted(x), sorted(y_predict_regression_ridge.ravel()), '-y', label='regression: ridge')
plt.plot(sorted(x), sorted(y_predict_scikit.ravel()), 'g', label='Scikit')
plt.legend()
plt.show()

"""
Make convergenceplot:
"""

plt.plot(range(len(MSE_gdmom_ridge)), MSE_gdmom_ridge, '-m', label='GD momentum: Ridge')
plt.plot(range(len(MSE_gdmom_ols)), MSE_gdmom_ols, 'r-', label='GD momentum: ols')
plt.plot(range(len(MSE_gd_ols)), MSE_gd_ols, 'b-', label='plain GD: ols')
plt.plot(range(len(MSE_gd_ridge)), MSE_gd_ridge, 'g-', label='plain GD: ridge')
plt.axis([0,max_iter,0,15])
plt.title('Convergence comparison GD wit and without momentum')
plt.ylabel('MSE scores')
plt.xlabel('Number of iterations')
plt.legend()
plt.show()

"""
Comment:  can make step size educated if you want, see that it statrs on a lower 
MSE score. 

Plain GD ridge and GD-momentum ols is performing fastest, with GD plain ridge having a lower 
mse score from the beginning. 

Should make a plot of MSE scores versus different step sizes to determine a better 
step size!

parameters: lambda, step_size, max_iter, tol, educated/not educated guess...

make table to show difference in final MSE score for different methods? how far they deviate
from the "perfect" score obtained by the regression method?
"""