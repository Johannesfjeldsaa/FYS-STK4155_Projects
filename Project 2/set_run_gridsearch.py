from Gridsearch import GridSearch_LinReg_epochs_batchsize
import numpy as np
from LinRegression import LinRegression
from dataframes import df_analysis_method_is_index

## set up data
np.random.seed(1997)
n = 100
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


optimization_methods = ['no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam']
results_header = ['batchsize MSE', 'epochs MSE', 'batchsize R2', 'epochs R2']

regression_method = 'OLS'
approach = 'analytical'

batch_epoch_analysis_dict = {}

for optimization in optimization_methods:
    MSE, R2 = GridSearch_LinReg_epochs_batchsize(poly_degree, x, y, initial_step,
                                                               batch_size,
                                                               epochs, lmbda=0,
                                                               regression_method=regression_method,
                                                               approach=approach,
                                                               optimization=optimization, momentum=momentum)
    print(f'SGD batch-epoch analysis: regression method: {regression_method}, approach: {approach}, optimization: {optimization}')
    i, j = np.where(MSE == np.min(MSE))
    batch_size_MSE = batch_size[i][0]
    epochs_MSE = epochs[j][0]

    print("Optimal batch size MSE = ", batch_size_MSE)
    print("Optimal epochs MSE = ", epochs_MSE)

    k, l = np.where(R2 == np.max(R2))
    batch_size_R2 = batch_size[k][0]
    epochs_R2 = epochs[l][0]

    print("Optimal batch size R2 = ", batch_size_R2)
    print("Optimal epochs R2 = ", epochs_R2)

    batch_epoch_analysis_dict[optimization] = [batch_size_MSE, epochs_MSE, batch_size_R2, epochs_R2]

print(batch_epoch_analysis_dict)

df = df_analysis_method_is_index(batch_epoch_analysis_dict, optimization_methods, results_header)

print(df)
