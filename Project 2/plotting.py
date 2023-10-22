""""
Draft of plotting functins (to become class later)

"""

# draft for comparing predicted model and convergencegraph

# for run OLS, analytical, SGD, all methods . collect betas and MSE_scores
optimization_methods = ['no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam']
analysis_dict = {'no momentum': [70, 700, 100, 300], 'momentum': [5, 200, 5, 300], 'RMSprop': [40, 700, 20, 100], 'adagrad': [40, 500, 100, 100], 'adam': [70, 1000, 70, 1000]}



for key, value in analysis_dict.items():
    beta, MSE = SGD(linreg.X, linreg.y, value[0], value[1], 0.1, 0.3,
                    10**-8, regression_method='OLS', optimization=key)
    y_pred = linreg.X@beta



    plt.plot(sorted(x), sorted(y_pred), label=key)

    #plt.plot(range(len(MSE))[::1000], MSE[::1000], label=key)  # prøvde på en konvergensgraf

plt.plot(x[::15],y[::15],'ro')
plt.title('Convergencegraph: analytical SGD analysis for OLS')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.show()

