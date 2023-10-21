import numpy as np
from LinRegression import LinRegression

"""
Class for Stochastic gradient descent  (make it to a class soon)

This method include stochastic gradient with following methods:
    momentum (without moment, choose momentum = 0)
    adagrad
    rmsprop
    adam
 
Before choosing optimization method, regression method has to be put in. Existing regression methods:
    OLS
    Ridge
"""



# inspired by daniel haas code, found in his repository for project 2. (spørr om ok=

def SGD(X, y, batch_size, n_epochs, initial_step, momentum, regression_method=None,
        lmbd=None, optimization=None, rho1=None, rho2=None):

    num_batch = int(len(y) / batch_size)  # num. batches
    p = X.shape[1]  # number of features

    beta = np.random.randn(p, 1)  # initialize beta
    delta = 1e-8  # to avoid division by zero: relevant for adagrad, RMSprop and adam.

    def learning_schedule(k):
        # providing updated and decayed learning rate, based on the current iteration.
        alpha = k / (n_epochs * num_batch)  # number of next iteration, divided by the total number of iterations set up
        new_learning_rate = (1-alpha)*initial_step+ alpha*initial_step*0.01  # E_t should be set to 1 % of initial guess
        return new_learning_rate

    change_vector = np.zeros((p, 1))  # initiate vector used for computing change in beta

    # step_size = initial_step_guess no in use

    ind = np.arange(len(y))  # indice by the lenghth of the vector

    for epoch in range(1, n_epochs + 1):
        s = np.zeros((p, 1))  # for computing first and second moments: adam
        r = np.zeros(shape=(p, p)) # for computing first and second moments: adagrad, adam, RMSprop
        random_ind = np.random.choice(ind, replace=False,
                                          size=ind.size)  # shuffle the data, without replacement
        batches = np.array_split(random_ind, num_batch)  # split into batches

        for i in range(num_batch):

            #Pick out belonging design matrix and y values to the randomized batch
            X_batch = X[batches[i], :]
            y_batch = y[batches[i]]

            # Compute the gradient using the data in minibatch k
            if regression_method == 'Ridge':
                gradient = 2.0 / batch_size * X_batch.T @ (X_batch @ beta - y_batch) + 2.0 * lmbd * beta
            elif regression_method == 'OLS':
                gradient = (2.0 / batch_size) * X_batch.T @ (X_batch @ beta - y_batch)
            else:
                raise Exception('no valid regression_method given!')

            step_size = learning_schedule(k=epoch * num_batch + i)  # linearly decaying learning rate (as described in Goodfellow)

            if optimization is None:
                change_vector = -step_size * gradient + momentum * change_vector

            elif optimization == "adagrad":
                r = r + gradient @ gradient.T  # sum of squared gradients
                rr = np.diagonal(r)  # we want g_i**2
                scale = np.c_[1 / (delta + np.sqrt(rr))]
                change_vector = -step_size * np.multiply(scale,
                                            gradient) + momentum * change_vector  # scale gradient element-wise

            elif optimization == "RMSprop":
                r = rho1 * r + (1 - rho1) * gradient @ gradient.T
                rr = np.c_[np.diagonal(r)]
                scale = np.c_[1.0 / (delta + np.sqrt(rr))]
                change_vector = -step_size * np.multiply(scale, gradient)  # scale gradient element-wise

            elif optimization == "adam":
                t = i + 1  # iteration number
                # here we compute 1st and 2nd moments
                r = rho2 * r + (1 - rho2) * gradient @ gradient.T
                s = rho1 * s + (1 - rho1) * gradient

                ss = s / (1 - rho1 ** t)  # here we correct the bias
                rr = np.c_[np.diagonal(r) / (1 - rho2 ** t)]

                change_vector = np.c_[-step_size * ss / (delta + np.sqrt(rr))]   # scale gradient element-wise

            else:
                raise Exception("Invalid optimization method.")

            beta += change_vector

    return beta  #returning the optimal beta from the optimization method used
