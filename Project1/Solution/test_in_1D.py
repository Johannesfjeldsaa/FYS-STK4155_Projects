from OLS import OLS

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


np.random.seed(200)
n = 100
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + np.random.normal(0, 0.1, x.shape)


# Plotting to get an idea of what we are dealing with.
fig, ax = plt.subplots(figsize=[8, 6])

ax.scatter(x, y,
           color='blue',
           label='Random data')

plt.show()


ols = OLS(5, x, y)

print(np.shape(ols.X))


X_train, X_test, y_train, y_test = ols.split_data(.4)

print(f'Split performed: {ols.splitted}')

beta = ols.train_by_OLS(train_on_scaled=False)


y_tilde = ols.predict_training()
y_pred = ols.predict_test()

(print(f'y_train: {np.shape(ols.y_train)}'))
print(f'y_tilde: {np.shape(y_tilde)}')
print(f'y_test: {np.shape(ols.y_test)}')
print(f'y_pred: {np.shape(y_pred)}')


# The mean squared error
print(f'Mean squared error training: {ols.MSE(ols.y_train, ols.y_pred_train):.4f}')
print(f'Mean squared error test: {ols.MSE(ols.y_test, ols.y_pred_test):.4f}')


# Plotting to get an idea of what we are dealing with.
fig, ax = plt.subplots(figsize=[8, 6])

ax.scatter(x, y,
           color='blue',
           label='Random data')


sorted_indices = np.argsort(ols.X_test[:, 1])  # Get the indices that would sort ols.X_test[:, 1] in ascending order
sorted_x_test = ols.X_test[:, 1][sorted_indices]  # Sort ols.X_test[:, 1] in ascending order
sorted_y_pred_test = ols.y_pred_test[sorted_indices]  # Apply the same sorting to ols.y_pred_test

ax.plot(sorted_x_test, sorted_y_pred_test,
           color='red',
           label='predicted data')


plt.show()
