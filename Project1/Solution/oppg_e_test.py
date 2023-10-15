import numpy as np
import matplotlib.pyplot as plt
from Franke_function import FrankeFunction
from LinRegression import LinRegression

"""
Perform then a bias-variance analysis of franke-function 
function by studying the MSE value as function of the complexity of your model.
Use ordinary least squares only.
"""

np.random.seed(1)
n= 100
n_boostraps = 100

x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
z = FrankeFunction(x,y)
poly_order = 16


error_liste = np.zeros(poly_order)
bias_liste = np.zeros(poly_order)
variance_liste = np.zeros(poly_order)
polydegree_liste = np.zeros(poly_order)


for degree in range(poly_order):

    bootstrapping = LinRegression(degree, x, y, z)
    bootstrapping.split_data(1 / 5)
    y_pred, error, bias, variance = bootstrapping.bootstrapping_train_model(n_boostraps)

    polydegree_liste[degree] = degree
    error_liste[degree] = error
    bias_liste[degree] = bias
    variance_liste[degree] = variance

    print('Polynomial degree:', degree)
    print('Error:', error_liste[degree])
    print('Bias^2:', bias_liste[degree])
    print('Var:', variance_liste[degree])
    print('{} >= {} + {} = {}'.format(error_liste[degree], bias_liste[degree], variance_liste[degree], bias_liste[degree]+variance_liste[degree]))

plt.figure()
plt.plot(polydegree_liste, error_liste, label='Error')
plt.plot(polydegree_liste, bias_liste, label='bias')
plt.plot(polydegree_liste, variance_liste, label='Variance')
plt.legend()
plt.show()
