import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from cost_functions import *
import jax
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler

# Load the data
cancer = load_breast_cancer()
X = cancer.data
target = cancer.target
print(target)
target = target.reshape(target.shape[0], 1)
print(target)

X_train, X_test, y_train, y_test = train_test_split(X, target) #random_state=0)
print(X_train.shape)
print(X_test.shape)

#now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Making a data frame
cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Now i want to find the target output from NN to set this for plotting and comparing to logreg
# follow rest of log reg example after this has been shown.
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]


"""
starting to get out what i want
"""

# inputs to NN:
input_nodes = X_train.shape[1]
output_nodes = 1



cross_entropy_func = CostCrossEntropy
cross_entropy_derivative = jax.grad(cross_entropy_func(target))

# både cost function og cross entry derivate må kunne sendes inn i backprop

