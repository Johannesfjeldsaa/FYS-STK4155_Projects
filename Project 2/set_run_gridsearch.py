from Gridsearch import GridSearch_LinReg_epochs_batchsize
import numpy as np
from LinRegression import LinRegression

## set up data
np.random.seed(1997)
n = 200
x = 0.5*np.random.rand(n, 1)
y = 4 + 3*x + 2*x*x + np.random.randn(n,1)
poly_degree = 2
linreg = LinRegression(poly_degree, x, y)

#parameters
#max_iter = 500
lambd = 0.1
tol = 10**-8
momentum = 0.3
initial_step = 0.1

#testing diff combos of batch and n epochs
batch_size = np.array([5, 10, 20, 40, 70, 100])
epochs = np.array([100, 200, 300, 500, 700, 1000])

#lists for storing to make df
methods = ['no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam']
results_header = ['batchsize MSE', 'epochs MSE', 'MSE score', 'batchsize R2', 'epochs R2', 'R2 score']

# OLS analytical, no momentum
MSE, R2 = GridSearch_LinReg_epochs_batchsize(poly_degree, x, y, initial_step, batch_size,
                                                epochs, lmbda=0, regression_method='OLS',
                                                approach='analytical', optimization=None, momentum=0)
print('OLS analytical SGD, no momentum')
a, b = np.where(MSE == np.min(MSE))
opt_batch_size_MSE = batch_size[a][0]
opt_epochs_MSE = epochs[b][0]

print("Optimal batch size MSE = ", opt_batch_size_MSE)
print("Optimal epochs MSE = ", opt_epochs_MSE)

c, d = np.where(R2 == np.max(R2))
opt_batch_size_R2 = batch_size[c][0]
opt_epochs_R2 = epochs[d][0]

print("Optimal batch size R2 = ", opt_batch_size_R2)
print("Optimal epochs R2 = ", opt_epochs_R2)


# OLS analytical momentum
MSE, R2 = GridSearch_LinReg_epochs_batchsize(poly_degree, x, y, initial_step, batch_size,
                                                epochs, lmbda=0, regression_method='OLS',
                                                approach='analytical', optimization='momentum', momentum=momentum)

print('OLS analytical SGD, momentum')
f, g = np.where(MSE == np.min(MSE))
opt_batch_size_MSE = batch_size[f][0]
opt_epochs_MSE = epochs[g][0]

print("Optimal batch size MSE = ", opt_batch_size_MSE)
print("Optimal epochs MSE = ", opt_epochs_MSE)

h, i = np.where(R2 == np.max(R2))
opt_batch_size_R2 = batch_size[h][0]
opt_epochs_R2 = epochs[i][0]

print("Optimal batch size R2 = ", opt_batch_size_R2)
print("Optimal epochs R2 = ", opt_epochs_R2)


# OLS analytical RMS_prop
MSE, R2 = GridSearch_LinReg_epochs_batchsize(poly_degree, x, y, initial_step, batch_size,
                                                epochs, lmbda=0, regression_method='OLS',
                                                approach='analytical', optimization='RMSprop', momentum=momentum)
print('OLS analytical SGD, RMS_prop')
j, k = np.where(MSE == np.min(MSE))
opt_batch_size_MSE = batch_size[j][0]
opt_epochs_MSE = epochs[k][0]

print("Optimal batch size MSE = ", opt_batch_size_MSE)
print("Optimal epochs MSE = ", opt_epochs_MSE)

l, m = np.where(R2 == np.max(R2))
opt_batch_size_R2 = batch_size[l][0]
opt_epochs_R2 = epochs[m][0]

print("Optimal batch size R2 = ", opt_batch_size_R2)
print("Optimal epochs R2 = ", opt_epochs_R2)

# OLS analytical adagrad
MSE, R2 = GridSearch_LinReg_epochs_batchsize(poly_degree, x, y, initial_step, batch_size,
                                                epochs, lmbda=0, regression_method='OLS',
                                                approach='analytical', optimization='adagrad', momentum=momentum)

print('OLS analytical SGD, adagrad')
n, o = np.where(MSE == np.min(MSE))
opt_batch_size_MSE = batch_size[n][0]
opt_epochs_MSE = epochs[o][0]

print("Optimal batch size MSE = ", opt_batch_size_MSE)
print("Optimal epochs MSE = ", opt_epochs_MSE)

p, q = np.where(R2 == np.max(R2))
opt_batch_size_R2 = batch_size[p][0]
opt_epochs_R2 = epochs[q][0]

print("Optimal batch size R2 = ", opt_batch_size_R2)
print("Optimal epochs R2 = ", opt_epochs_R2)

# OLS analytical momentum
MSE, R2 = GridSearch_LinReg_epochs_batchsize(poly_degree, x, y, initial_step, batch_size,
                                                epochs, lmbda=0, regression_method='OLS',
                                                approach='analytical', optimization='adam', momentum=momentum)


print('OLS analytical SGD, adam')
r, s = np.where(MSE == np.min(MSE))
opt_batch_size_MSE = batch_size[r][0]
opt_epochs_MSE = epochs[s][0]

print("Optimal batch size MSE = ", opt_batch_size_MSE)
print("Optimal epochs MSE = ", opt_epochs_MSE)

t, u = np.where(R2 == np.max(R2))
opt_batch_size_R2 = batch_size[t][0]
opt_epochs_R2 = epochs[u][0]

print("Optimal batch size R2 = ", opt_batch_size_R2)
print("Optimal epochs R2 = ", opt_epochs_R2)
