import autograd.numpy as np
from autograd import grad

""""
Perform exercise 41 and 42 with automatic differentiation
"""

"""
Class for Stochastic gradient descent  (make it to a class soon). When making it a class, the
MSE scores can be stored to the class object. 

This method include stochastic gradient with following methods:
    momentum (without moment, choose momentum = 0)
    adagrad
    rmsprop
    adam

Before choosing optimization method, regression method has to be put in. Existing regression methods:
    OLS
    Ridge
"""


# kan sikkert helle legge GD inn i SGD? blir mye "to ganger". ellers så er det kanskje kjekt å
# kunne kalle på GD eller SGD utenfr klassen, for oversikt

def AD_GD(X, y, initial_step, max_iter, tol, lambd=None, regression_method=None, momentum=None,
       optimization=None, rho1=None, rho2=None):
    """
    No epochs and no batches
    :param X:
    :param y:
    :param step_size:
    :param max_iter:
    :param lambd:
    :param regression_method:
    :return: beta: optimal betavalue for the chosen regression and optimization model
    """
    n = X.shape[0]
    p = X.shape[1]
    delta = 10**-8

    beta = np.random.randn(p, 1)  # initial guess beta
    change_vector = np.zeros((p, 1))  # initialize velocity vector (p x 1), change vector, to be updated

    # learning schedule for decaying rate when applying momentum as optimization method
    def learning_schedule(k):
        # providing updated and decayed learning rate, based on the current iteration.
        alpha = k / (max_iter)  # number of next iteration, divided by the total number of iterations set up
        new_learning_rate = (1 - alpha) * initial_step + alpha * initial_step * 0.01  # E_t should be set to 1 % of initial guess
        return new_learning_rate

    step_size = initial_step  # for adagrad, adam, and rmsprop??? sjekk om sant? lurer på grunn av høres ut som adam, adagrad and rmsprop hører til tuner sin egen ut fra oppg tekst

    # Applying MSE as loss function

    def loss_function_no_momentum_ols(beta):
        return (1/n) * np.sum((y - X @ beta) ** 2)

    def loss_function_no_momentum_ridge(beta, lambd):
        y_pred = X @ beta  # predicted response (n x 1)
        sse = (np.sum((y - y_pred) ** 2)) / n  # sum of squared errors
        reg = lambd * np.sum(beta ** 2)  # regularization term
        return sse + reg

    #usikker på om disse to under trengs når det ikke er stochastic? er det i boka til morten for vanlig GD med momentum og
    def loss_function_momentum_ols(X,y,beta):
        return np.sum((y - X@beta)**2) / n
    def loss_function_momentum_ridge(X,y,beta,lambd):
        y_pred = X @ beta  # predicted response (n x 1)
        sse = (np.sum((y - y_pred) ** 2)) / n  # sum of squared errors
        reg = lambd * np.sum(beta ** 2)  # regularization term
        return sse + reg

    training_grad_no_momentum_ols = grad(loss_function_no_momentum_ols)
    training_grad_momentum_ols = grad(loss_function_momentum_ols,2) # define which parameter is the objective for the gradient
    training_grad_no_momentum_ridge = grad(loss_function_no_momentum_ridge)
    training_grad_momentum_ridge = grad(loss_function_momentum_ridge,2)  # define which parameter is the objective for the gradient

    MSE_scores = []

    for iteration in range(max_iter):
        s = np.zeros((p, 1))  # for computing first and second moments: adam
        r = np.zeros(shape=(p, p))  # for computing first and second moments: adagrad, adam, RMSprop

        if regression_method == 'OLS':
            gradient_no_mom = training_grad_no_momentum_ols(beta)
            gradient_mom = training_grad_momentum_ols(X, y, beta)

            if np.linalg.norm(gradient_no_mom.any() or gradient_mom.any()) < tol:  # check convergence criterion
                print("Converged!")
                break

        elif regression_method == 'Ridge':
            gradient_no_mom = training_grad_no_momentum_ridge(beta)
            gradient_mom = training_grad_momentum_ridge(X, y, beta)

            if np.linalg.norm(gradient_no_mom.any() or gradient_mom.any()) < tol:  # check convergence criterion
                print("Converged!")
                break
        else:
            raise Exception('No valid regression method given')

        if optimization == 'no momentum':
            step_size = learning_schedule(iteration)
            beta -= step_size * gradient_no_mom

        elif optimization == 'momentum':
            step_size = learning_schedule(iteration)
            change_vector = momentum * change_vector - step_size * gradient_mom

        elif optimization == 'adagrad':
            r = r + gradient_mom @ gradient_mom.T  # sum of squared gradients
            rr = np.diagonal(r)  # we want g_i**2
            scale = np.c_[1 / (delta + np.sqrt(rr))]
            change_vector = -step_size * np.multiply(scale,
                                                     gradient_mom) + momentum * change_vector  # scale gradient element-wise

        elif optimization == 'adam':
            t = iteration + 1  # iteration number
            # here we compute 1st and 2nd moments
            r = rho2 * r + (1 - rho2) * gradient_mom @ gradient_mom.T
            s = rho1 * s + (1 - rho1) * gradient_mom

            ss = s / (1 - rho1 ** t)  # here we correct the bias
            rr = np.c_[np.diagonal(r) / (1 - rho2 ** t)]

            change_vector = np.c_[
                -step_size * ss / (delta + np.sqrt(rr))]  # scale gradient element-wise

        elif optimization == 'RMSprop':
            r = rho1 * r + (1 - rho1) * gradient_mom @ gradient_mom.T
            rr = np.c_[np.diagonal(r)]
            scale = np.c_[1.0 / (delta + np.sqrt(rr))]
            change_vector = -step_size * np.multiply(scale,
                                                     gradient_mom)  # scale gradient element-wise

        else:
            raise Exception('No valid optimization method given')

        beta += change_vector
        MSE_scores.append(((y - X @ beta) ** 2) / n)

    return beta, MSE_scores


# inspired by daniel haas code, found in his repository for project 2. (spørr om ok?)
def AD_SGD(X, y, batch_size, n_epochs, initial_step, momentum, tol, regression_method=None,
        lmbd=None, optimization=None, rho1=None, rho2=None):

    n = X.shape[0]
    p = X.shape[1]  # number of features
    num_batch = int( n / batch_size)  # num. batches

    beta = np.random.randn(p, 1)  # initialize beta
    delta = 1e-8  # to avoid division by zero: relevant for adagrad, RMSprop and adam.

    def learning_schedule(k):
        # providing updated and decayed learning rate, based on the current iteration.
        alpha = k / (n_epochs * num_batch)  # number of next iteration, divided by the total number of iterations set up
        new_learning_rate = (1 - alpha) * initial_step + alpha * initial_step * 0.01  # E_t should be set to 1 % of initial guess
        return new_learning_rate

    step_size = initial_step  # for adam, adagrad and RMSprop where they are tuning learning rate inside optimization
    change_vector = np.zeros((p, 1))  # initiate vector used for computing change in beta

    # choosing to divide by number of bacthes instead of n for the SGD automatic differentiation
    def loss_function_no_momentum_ols(beta):
        return (1/num_batch) * np.sum((y - X @ beta) ** 2)

    def loss_function_no_momentum_ridge(beta, lambd):
        y_pred = X @ beta  # predicted response (n x 1)
        sse = (np.sum((y - y_pred) ** 2)) / num_batch  # sum of squared errors
        reg = lambd * np.sum(beta ** 2)  # regularization term
        return sse + reg

    def loss_function_momentum_ols(X,y,beta):
        return (1/num_batch)*np.sum((y - X@beta)**2)
    def loss_function_momentum_ridge(X,y,beta,lambd):
        y_pred = X @ beta  # predicted response (n x 1)
        sse = (np.sum((y - y_pred) ** 2)) / num_batch  # sum of squared errors
        reg = lambd * np.sum(beta ** 2)  # regularization term
        return sse + reg

    training_grad_no_momentum_ols = grad(loss_function_no_momentum_ols)
    training_grad_momentum_ols = grad(loss_function_momentum_ols,2) # define which parameter is the objective for the gradient
    training_grad_no_momentum_ridge = grad(loss_function_no_momentum_ridge)
    training_grad_momentum_ridge = grad(loss_function_momentum_ridge,2)  # define which parameter is the objective for the gradient

    ind = np.arange(len(y))  # indice by the lenghth of the vector

    MSE_scores = []

    for epoch in range(1, n_epochs + 1):
        s = np.zeros((p, 1))  # for computing first and second moments: adam
        r = np.zeros(
            shape=(p, p))  # for computing first and second moments: adagrad, adam, RMSprop
        random_ind = np.random.choice(ind, replace=False,
                                      size=ind.size)  # shuffle the data, without replacement
        batches = np.array_split(random_ind, num_batch)  # split into batches

        for i in range(num_batch):

            # Pick out belonging design matrix and y values to the randomized batch
            X_batch = X[batches[i], :]
            y_batch = y[batches[i]]

            # Compute the gradient using the data in minibatch k
            if regression_method == 'Ridge':
                gradient_no_mom = training_grad_no_momentum_ridge(beta)
                gradient_mom = training_grad_momentum_ridge(X_batch, y_batch, beta)

                if np.linalg.norm(gradient_no_mom.any() or gradient_mom.any()) < tol:  # check convergence criterion
                    print("Converged!")
                    break

            elif regression_method == 'OLS':
                gradient_no_mom = training_grad_no_momentum_ols(beta)
                gradient_mom = training_grad_momentum_ols(X_batch, y_batch, beta)

                if np.linalg.norm(gradient_no_mom.any() or gradient_mom.any()) < tol:  # check convergence criterion
                    print("Converged!")
                    break

            else:
                raise Exception('no valid regression_method given!')

            if optimization == 'no momentum':
                step_size = learning_schedule(k=epoch * num_batch + i)
                change_vector = -step_size * gradient_no_mom

            if optimization == 'momentum':
                step_size = learning_schedule(k=epoch * num_batch + i)  # linearly decaying learning rate (as described in Goodfellow)
                change_vector = -step_size * gradient_mom + momentum * change_vector

            elif optimization == "adagrad":
                r = r + gradient_mom @ gradient_mom.T  # sum of squared gradients
                rr = np.diagonal(r)  # we want g_i**2
                scale = np.c_[1 / (delta + np.sqrt(rr))]
                change_vector = -step_size * np.multiply(scale,
                                                         gradient_mom) + momentum * change_vector  # scale gradient element-wise

            elif optimization == "RMSprop":
                r = rho1 * r + (1 - rho1) * gradient_mom @ gradient_mom.T
                rr = np.c_[np.diagonal(r)]
                scale = np.c_[1.0 / (delta + np.sqrt(rr))]
                change_vector = -step_size * np.multiply(scale,
                                                         gradient_mom)  # scale gradient element-wise

            elif optimization == "adam":
                t = i + 1  # iteration number
                # here we compute 1st and 2nd moments
                r = rho2 * r + (1 - rho2) * gradient_mom @ gradient_mom.T
                s = rho1 * s + (1 - rho1) * gradient_mom

                ss = s / (1 - rho1 ** t)  # here we correct the bias
                rr = np.c_[np.diagonal(r) / (1 - rho2 ** t)]

                change_vector = np.c_[
                    -step_size * ss / (delta + np.sqrt(rr))]  # scale gradient element-wise

            else:
                raise Exception("Invalid optimization method.")

            beta += change_vector
            MSE_scores.append(np.sum((y_batch - X_batch @ change_vector) ** 2) / batch_size)

    return beta, MSE_scores  # returning the optimal beta from the optimization method used
