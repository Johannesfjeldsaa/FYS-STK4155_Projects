from NeuralNetwork import Neural_Network
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import Neural_Network


#COST FUNCTION breast cancer

def CostCrossEntropy(target):
    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

def analytic_derivate_CostCross(activation_output, target):
    dCW_da = (activation_output - target) / (activation_output * (1 - activation_output))
    return dCW_da


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

"""
This code from Mortens online book is showing results from only logistic regression.
Can be used to compare with for task E. 

"""

"""
# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,y_test)))

#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Logistic Regression
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))

# Making a data frame
cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)

fig, axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:,i], bins =50)
    ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc ="best")
fig.tight_layout()
plt.show()

import seaborn as sns
correlation_matrix = cancerpd.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
plt.figure(figsize=(15,8))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

"""


"""
our try on breast cancer
"""

# Load the data
cancer = load_breast_cancer()
X = cancer.data
target_true = cancer.target
target_true = target_true.reshape(target_true.shape[0], 1)



"""
#splitting
X_train, X_test, y_train, y_test = train_test_split(X, target) #random_state=0)

#now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""

# NN setup
n_hidden_layers = 1
n_hidden_nodes = 2
n_outputs = 1
learning_rate=0.1


#training NN with breastcancer data
NN_output_cancer = []

for X_row, target_row in zip(X, target_true):

    X = jnp.array([X_row])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                      learning_rate, activation_function='sigmoid',
                      classification_problem=True)

    nn.feed_forward()


    nn.feed_backward()
    nn.train()

    NN_output_cancer.append(nn.output_layer.output)


print(NN_output_cancer)
print(len(target_true))

print(len(NN_output_cancer))



"""
# Now i want to find the target output from NN to set this for plotting and comparing to logreg
# follow rest of log reg example after this has been shown.
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
"""

"""
starting to get out what i want
"""

# Making a data frame
#cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)




"""
#%%
np.random.seed(1234)

# Create design matrix
X_orig = jnp.array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])


# The XOR gate
target_XOR = jnp.array( [ 0, 1 ,1, 0])
# The OR gate
target_OR = jnp.array( [ 0, 1 ,1, 1])
# The AND gate
target_AND = jnp.array( [ 0, 0 ,0, 1])

n_hidden_layers = 1
n_hidden_nodes = 2
n_outputs = 1
learning_rate=0.1

# AND gate
NN_output_AND = []

for X_row, target_row in zip(X_orig, target_AND):

    X = jnp.array([X_row])
    target = jnp.array([target_row])

    ffnn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                      learning_rate, activation_function='sigmoid',
                      classification_problem=True)

    ffnn.feed_forward()


    ffnn.feed_backward()
    ffnn.train()

    NN_output_AND.append(ffnn.output_layer.output)


print(NN_output_AND)
"""

def accuracy_test(y_true, y_pred):
    # y_true is the true labels of the data
    # y_pred is the predicted labels of the data by the neural network
    # Both y_true and y_pred are numpy arrays of the same shape

    # Compare y_true and y_pred element-wise and count the number of matches
    matches = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            matches = matches + 1

    # Calculate the accuracy as the ratio of matches to the total number of data points
    accuracy = matches / len(y_true)

    # Return the accuracy as a percentage
    return accuracy * 100

## test your accuracy:
#accuracy_AND_gate = accuracy_test(target_AND, NN_output_AND)
#print(accuracy_AND_gate)

accuracy_breast_cancer = accuracy_test(target_true, NN_output_cancer)
print(accuracy_breast_cancer)