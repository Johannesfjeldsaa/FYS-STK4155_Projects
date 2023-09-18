import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Frankefunction import FrankeFunction
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
Part a) OLS on Franke function. Reuse from exercises w35 + w36

generate dataset with and without added stochastic noise from normal dist
Split data in training and test (Training= 4/5)
Scaling/center data   "include in discussion"
OLS with matrix inversion ( own code )
Evaluate MSE + R^2 score
Plot MSE + R^2 score as function of polynomial degree (up to five)
Plot parameters beta as polynomial order increases
"""

# Function for finding B_ols
def find_beta_ols(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Function for calculating MSE-score
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2_score(y_data, y_model):
    n = np.size(y_model)
    y_sample_mean = np.sum(y_data)/n
    return (1 - (np.sum((y_data-y_model)**2)) / (np.sum((y_data-y_sample_mean)**2)))

np.random.seed(1)
n = 100
degree = 5

x = np.linspace(-3, 3, n).reshape(-1, 1)  # 1 column with n rows
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape) #add noise later, make sure its in the normal dist

# Making a function to calculate MSE for different complexities.

def MSE_R2_score_ols_analysis(polynomial_degree,x,y):
    X_list = []
    for polynomial in range(polynomial_degree+1):
        if polynomial != 0:
            poly = PolynomialFeatures(degree=polynomial, include_bias=False) # leave out intercept
            X_list.append(poly.fit_transform(x[:,0][:, np.newaxis]))

    #Perform OLS analysis on the different design matrices with different polynomial degrees

    MSE_ols = []
    R2_ols = []
    beta_parameters = []

    for design_matrix in X_list:

        # Implementing test
        identity_matrix = np.eye(np.size(design_matrix,1),np.size(design_matrix,1))
        A = design_matrix @ np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T

        if A.all() == identity_matrix.all():
            print('design matrix is equal to the identity matrix, the mean squared error should be 0')
        else:
            print('The design matrix is not equal to the identity matrix')

        X_train, X_test, y_train, y_test = train_test_split(design_matrix, y, test_size=0.25)

        # Centering the data
        X_train_mean = np.mean(X_train, axis=0)
        X_train_scaled = X_train - X_train_mean
        X_test_scaled = X_test - X_train_mean
        y_scaler = np.mean(y_train)
        y_train_scaled = y_train - y_scaler

        # Writing own code for OLS with matrix inversion to find beta_ols
        beta_ols = find_beta_ols(X_train_scaled,y_train_scaled)
        beta_parameters.append(beta_ols)
        y_predicted = X_test_scaled@beta_ols + y_scaler
        MSE_ols.append(MSE(y_test, y_predicted))
        R2_ols.append(R2_score(y_test, y_predicted))


    return MSE_ols, R2_ols, beta_parameters

MSE_scores, R2_scores, beta_params = MSE_R2_score_ols_analysis(degree,x,y)

"""
spm: nå har jeg sentrert alt som går inn i modellen og ikke tatt med intercept selv om det
kun er OLS som blir gjennomført. Er dette vits? mtp på hva som skal inn i diskusjonen

Jeg tror dette skal tas bort, i og med at dataene er skalert mellom [0,1] til frankefunksjonen, 
slik det har blitt definert for oss. kanskje det må tas med når man skal gå gjennom den ekte dataen?
"""

# Plotting of R2 and MSE scores against polynomial order
fig, ax = plt.subplots()
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Predicted mean square error')
ax.plot(np.arange(1,degree+1), MSE_scores, alpha=0.7, lw=2, color='r', label='MSE score')
ax2 = ax.twinx()
ax2.set_ylabel('Predicted R2 score')
ax2.plot(np.arange(1,degree+1), R2_scores, alpha=0.7, lw=2, label='R2 score')
ax.legend(loc='lower right')
ax2.legend(loc='center right')
plt.show()


# Plotting of beta parameters as a function of order
fig, ax = plt.subplots()
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Beta parameters')

for degree, beta_array in zip(np.arange(1, degree+1), beta_params):
    if degree == 1:
        ax.scatter(degree, beta_array[0,0], label='polynomial of order 1')
    else:
        ax.plot(np.arange(1,degree+1), beta_array[:,0], 'rx')
        ax.plot(np.arange(1, degree + 1), beta_array[:, 0], label=f'polynomial of order {degree}')
plt.legend()
plt.show()


"""
Now I am going to redo this code to be handling the frankefunction

i think its should be rewritten to not exclude the intercept and not senter it, as the data is
alrede scaled between  0 and 1

"""

# Make data. Here it is fixed values and fixed step size
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05) #+ np.random.normal(0, 0.1) #random stochastic noise
xv, yv = np.meshgrid(x,y, indexing='ij')  # prøvde å få den indeksert til matrise behandling
z = FrankeFunction(xv, yv)

def franke_MSE_R2_score_ols_analysis(polynomial_degree,z):
    X_list = []
    for polynomial in range(polynomial_degree+1):
        if polynomial != 0:
            poly = PolynomialFeatures(degree=polynomial, include_bias=False) # leave out intercept
            X_list.append(poly.fit_transform(z[:,0][:, np.newaxis]))

    #Perform OLS analysis on the different design matrices with different polynomial degrees

    MSE_ols = []
    R2_ols = []
    beta_parameters = []

    for design_matrix in X_list:
        X_train, X_test, y_train, y_test = train_test_split(design_matrix, z[0,:], test_size=0.25)

        # Centering the data
        X_train_mean = np.mean(X_train, axis=0)
        X_train_scaled = X_train - X_train_mean
        X_test_scaled = X_test - X_train_mean
        y_scaler = np.mean(y_train)
        y_train_scaled = y_train - y_scaler

        # Writing own code for OLS with matrix inversion to find beta_ols
        beta_ols = find_beta_ols(X_train_scaled,y_train_scaled)
        beta_parameters.append(beta_ols)
        y_predicted = X_test_scaled@beta_ols + y_scaler
        MSE_ols.append(MSE(y_test, y_predicted))
        R2_ols.append(R2_score(y_test, y_predicted))


    return MSE_ols, R2_ols, beta_parameters


franke_MSE_ols, franke_R2_ols, franke_beta_params = franke_MSE_R2_score_ols_analysis(degree,z)

# Plotting of R2 and MSE scores against polynomial order
fig, ax = plt.subplots()
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Predicted mean square error')
ax.plot(np.arange(1,degree+1), franke_MSE_ols, alpha=0.7, lw=2, color='r', label='MSE score')
ax2 = ax.twinx()
ax2.set_ylabel('Predicted R2 score')
ax2.plot(np.arange(1,degree+1), franke_R2_ols, alpha=0.7, lw=2, label='R2 score')
ax.legend(loc='lower right')
ax2.legend(loc='center right')
plt.show()

# Plotting of beta parameters as a function of order
fig, ax = plt.subplots()
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Beta parameters')

for degree, franke_betas in zip(np.arange(1, degree+1), franke_beta_params):
    if degree == 1:
        ax.scatter(degree, franke_betas, label='polynomial of order 1')
    else:
        ax.plot(np.arange(1,degree+1), franke_betas, 'rx')
        ax.plot(np.arange(1, degree + 1), franke_betas, label=f'polynomial of order {degree}')
plt.legend()
plt.show()

# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')  # Make poejction of 3d space down to a 2d fit
# Plot the surface.
surf = ax.plot_surface(xv, yv, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()