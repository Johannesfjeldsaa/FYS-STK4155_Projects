import os

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')

from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(200)
n = 100
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + np.random.normal(0, 0.1, x.shape)


# Plotting to get an idea of what we are dealing with.
fig, ax = plt.subplots(figsize=[8,6])

ax.scatter(x, y,
            color='blue',
            label='Random data')

# Legg til en legend nederst sentrert i en separat boks
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), #lokasjon
                   fancybox=True, shadow=True) # utseende for boksen
legend.get_frame().set_facecolor('white') #uteseende for boksenplt.show()

plt.show()



from OLS import OLS

ols = OLS(5, x, y)

print(np.shape(ols.X))


X_train, X_test, y_train, y_test = ols.split_data(.4)

print(f'Split performed: {ols.splitted}')

beta = ols.train_by_OLS(train_on_scaled=False)


y_tilde = ols.predict_traing()
y_pred = ols.predict_test()

(print(f'y_train: {np.shape(ols.y_train)}'))
print(f'y_tilde: {np.shape(y_tilde)}')
print(f'y_test: {np.shape(ols.y_test)}')
print(f'y_pred: {np.shape(y_pred)}')


# The mean squared error
print(f'Mean squared error training: {ols.MSE(ols.y_train, ols.y_pred_train):.4f}')
print(f'Mean squared error test: {ols.MSE(ols.y_test, ols.y_pred_test):.4f}')




# Plotting to get an idea of what we are dealing with.
fig, ax = plt.subplots(figsize=[8,6])

ax.scatter(x, y,
            color='blue',
            label='Random data')

ax.scatter(ols.X_test[:,1], ols.y_pred_test,
           color='red',
           label='predicted data')

# Legg til en legend nederst sentrert i en separat boks
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), #lokasjon
                   fancybox=True, shadow=True) # utseende for boksen
legend.get_frame().set_facecolor('white') #uteseende for boksenplt.show()

plt.show()
