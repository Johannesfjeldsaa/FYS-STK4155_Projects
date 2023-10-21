""""
Draft of plotting functins (to become class later)

"""

"""
Compare beta-param result and plot difference
"""
XT_X = linreg.X.T @ linreg.X
Id = n*lambd* np.eye(XT_X.shape[0])  #identitetsmatrise
beta_ridge = np.linalg.inv(XT_X+Id) @ linreg.X.T @ y
beta_ols = np.linalg.inv(linreg.X.T @ linreg.X) @ linreg.X.T @ y

#scikit
sgdreg = SGDRegressor(max_iter = max_iter, eta0=step_size, tol=tol, fit_intercept=False) # nå tas intercept med! "antatt at data er skalert"
sgdreg.fit(linreg.X, linreg.y.ravel())
intercept_scikit = sgdreg.intercept_
weights_scikit = np.array(sgdreg.coef_)
number_iterations = sgdreg.n_iter_
print('number iterations scikit:')
print(number_iterations)

y_predict_scikit = linreg.X @ weights_scikit
y_predict_ols = linreg.X @ beta_gd_mom_ols
y_predict_ridge = linreg.X @ beta_gd_mom_ridge
y_predict_regression_ols = linreg.X @ beta_ols
y_predict_regression_ridge = linreg.X @ beta_ridge

# eventuelt ta med scikit også..
plt.plot(x,y,'ro', label='actual data')
plt.plot(sorted(x), sorted(y_predict_ols.ravel()), 'b-', label='gradient: ols')
plt.plot(sorted(x), sorted(y_predict_ridge.ravel()), 'm-', label='gradient: ridge')# Blir det feil å sortere dem???
plt.plot(sorted(x), sorted(y_predict_regression_ols.ravel()), '-c', label='regression: ols')
plt.plot(sorted(x), sorted(y_predict_regression_ridge.ravel()), '-y', label='regression: ridge')
plt.plot(sorted(x), sorted(y_predict_scikit.ravel()), 'g', label='Scikit')
plt.legend()
plt.show()

"""
Make convergenceplot:
"""

plt.plot(range(len(MSE_gdmom_ridge)), MSE_gdmom_ridge, '-m', label='GD momentum: Ridge')
plt.plot(range(len(MSE_gdmom_ols)), MSE_gdmom_ols, 'r-', label='GD momentum: ols')
plt.plot(range(len(MSE_gd_ols)), MSE_gd_ols, 'b-', label='plain GD: ols')
plt.plot(range(len(MSE_gd_ridge)), MSE_gd_ridge, 'g-', label='plain GD: ridge')
plt.axis([0,max_iter,0,15])
plt.title('Convergence comparison GD wit and without momentum')
plt.ylabel('MSE scores')
plt.xlabel('Number of iterations')
plt.legend()
plt.show()