import numpy as np

# Create two 1-dimensional arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate them column-wise using np.c_
result = np.c_[a, b]

print(result)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create a grid of x and y values with 'ij' indexing
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y, indexing='ij')

# Flatten x and y into 1D arrays
x_flat = x.ravel()
y_flat = y.ravel()

print(x_flat.shape)
print(x_flat[:, np.newaxis].shape)
print(np.hstack((x_flat[:, np.newaxis], y_flat[:, np.newaxis])).shape)
poly_degree = 5

# Create the design matrix for the 2D polynomial using horizontal stacking
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

X = poly.fit_transform(np.hstack((x_flat[:, np.newaxis], y_flat[:, np.newaxis])))

print(X.shape)

