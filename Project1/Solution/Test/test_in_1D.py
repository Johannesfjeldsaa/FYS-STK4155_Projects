from LinRegression import LinRegression

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


Linreg = LinRegression(5, x, y)

print(np.shape(Linreg.X))


Linreg.split_data(.4)


print(f'Split performed: {Linreg.splitted}')

beta = Linreg.train_model(regression_method='OLS', train_on_scaled=False)


y_tilde = Linreg.predict_training()
y_pred = Linreg.predict_test()

(print(f'y_train: {np.shape(Linreg.y_train)}'))
print(f'y_tilde: {np.shape(y_tilde)}')
print(f'y_test: {np.shape(Linreg.y_test)}')
print(f'y_pred: {np.shape(y_pred)}')


# The mean squared error
print(f'Mean squared error training: {Linreg.MSE(Linreg.y_train, Linreg.y_pred_train):.4f}')
print(f'Mean squared error test: {Linreg.MSE(Linreg.y_test, Linreg.y_pred_test):.4f}')


# Plotting to get an idea of what we are dealing with.
fig, ax = plt.subplots(figsize=[8, 6])

ax.scatter(x, y,
           color='blue',
           label='Random data')


sorted_indices = np.argsort(Linreg.X_test[:, 1])  # Get the indices that would sort ols.X_test[:, 1] in ascending order
sorted_x_test = Linreg.X_test[:, 1][sorted_indices]  # Sort ols.X_test[:, 1] in ascending order
sorted_y_pred_test = Linreg.y_pred_test[sorted_indices]  # Apply the same sorting to ols.y_pred_test

ax.plot(sorted_x_test, sorted_y_pred_test,
           color='red',
           label='predicted data')


plt.show()
