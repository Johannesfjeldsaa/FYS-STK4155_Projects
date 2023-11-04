from Gridsearch import GridSearch_LinReg_epochs_batchsize
from dataframes import df_analysis_method_is_index
from plotting import plot_SGD_MSE_convergence_epoch_batch
from GD_class import *


"""
Functions 
"""

def cost_function_OLS(X, y, beta):
    n = len(y)  # Define the number of data points
    return (1.0/n) * jnp.sum((y - jnp.dot(X, beta))**2)

def analytical_gradient(X, y, beta):
    n = len(y)
    return (2.0/n)*jnp.dot(X.T, ((jnp.dot(X, beta))-y))

def make_design_matrix(x, degree):
    "Creates the design matrix for the given polynomial degree and ijnput data"

    X = np.zeros((len(x), degree + 1))

    for i in range(X.shape[1]):
        X[:, i] = np.power(x, i)

    return jnp.array(X)

"""
generate data and design matrix
"""

np.random.seed(1342)

true_beta = [2, 0.5, 3.2]

n = 1000

x = jnp.linspace(0, 1, n)
y = jnp.sum(jnp.asarray([x ** p * b for p, b in enumerate(true_beta)]),
                axis=0) + 0.2 * np.random.normal(size=len(x))

# Making a design matrix to use for linear regression part
degree = 2
X = make_design_matrix(x, degree)

# Set parameters
learning_rate = 0.1
tol=1e-3
momentum=0.5
delta= 1e-8
rho1 = 0.9
rho2 = 0.99

"""
make classes
"""


# OLS analytical Adam
grad_descentADAM = GradientDescentADAM(delta, rho1, rho2, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)

# OLS analytical RMSprop
grad_descentRMS_prop = GradientDescentRMSprop(delta, rho1, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)
# OLS analytical Adagrad
grad_descentAdagrad = GradientDescentAdagrad(delta, rho1, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)
# OLS analytical momentum
grad_descentMomentum = GradientDescentMomentum(momentum=0.3, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)

grad_descentNoMomentum = GradientDescentMomentum(momentum=0, X=X, y=y,
                                            learning_rate=learning_rate, tol=tol,
                                            cost_function=cost_function_OLS,
                                            analytic_gradient=analytical_gradient,
                                            skip_convergence_check=True,
                                            record_history=True)

"""
Perform gridsearch analysis
"""

#testing diff combos of batch and n epochs
num_batches = np.array([50, 40, 30, 20, 10, 5])
epochs = np.array([25, 50, 100, 200, 300, 500])

solution_dicts = []  # was intended if you wanted to provie other combinations, now only optimization analysis for SGD analytical
optimization_methods = ['no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam']
iteration_methods = ['stochastic']
approaches = ['analytical']

for i, optimization in enumerate(optimization_methods):
    sol_dict = {}
    for iteration, approach in zip(iteration_methods, approaches):
        sol_dict['optimization'] = optimization
        sol_dict['approach'] = approach
        sol_dict['iteration'] = iteration

    solution_dicts.append(sol_dict)

batch_epoch_analysis_dict = {}  # for making dataframe of results

for solution_dict in solution_dicts:

    optimization = solution_dict.get('optimization')

    betas = []
    for epoch, num_batch in zip(epochs, num_batches):

        if optimization == 'no momentum':
            beta = grad_descentNoMomentum.iterate(iteration_method="Stochastic",
                                            max_epoch=epoch,
                                            num_batches=num_batch)
            betas.append(beta)

        if  optimization == 'momentum':
            beta = grad_descentMomentum.iterate(iteration_method="Stochastic",
                                            max_epoch=epoch,
                                            num_batches=num_batch)
            betas.append(beta)

        if optimization == 'RMSprop':
            beta = grad_descentRMS_prop.iterate(iteration_method="Stochastic",
                                            max_epoch=epoch,
                                            num_batches=num_batch)
            betas.append(beta)

        if optimization == 'adagrad':
            beta = grad_descentAdagrad.iterate(iteration_method="Stochastic",
                                            max_epoch=epoch,
                                            num_batches=num_batch)
            betas.append(beta)

        if optimization == 'adam':

            beta = grad_descentADAM.iterate(iteration_method="Stochastic",
                                          max_epoch=epoch,
                                          num_batches=num_batch)
            betas.append(beta)


    MSE, R2, solution = GridSearch_LinReg_epochs_batchsize(betas, X, y, num_batches, epochs, solution_dict)

    print(f'Batch-epoch analysis for stochastic, analytical, {optimization} method')

    i, j = np.where(MSE == np.min(MSE))
    num_batch_MSE = num_batches[i][0]
    epochs_MSE = epochs[j][0]

    k, l = np.where(R2 == np.max(R2))
    num_batch_R2 = num_batches[k][0]
    epochs_R2 = epochs[l][0]

    min_MSE_score = np.min(MSE)
    max_R2_score = np.max(R2)

    print(f'Lowest MSE score obtained is {min_MSE_score}')
    print(f'Highest R2 score obtained is {max_R2_score}')
    print(f'The optimal number of batches and epoch for analysis of {solution} is {num_batch_MSE} '
          f'and {epochs_MSE} respectively for lowest MSE score ')
    print(f'The optimal number of batches and epoch for analysis of {solution} is {num_batch_R2} '
          f'and {epochs_R2} respectively for highest R2 score ')

    batch_epoch_analysis_dict[optimization] = [num_batch_MSE, epochs_MSE, num_batch_R2,
                                               epochs_R2, min_MSE_score, max_R2_score]



results_header = ['Num. batches MSE', 'Num. epochs MSE', 'Num. batches R2', 'Num. epochs R2',
                  'Min MSE score', 'Max R2 score']


df = df_analysis_method_is_index(batch_epoch_analysis_dict, optimization_methods, results_header)

print(batch_epoch_analysis_dict)
print(df)

