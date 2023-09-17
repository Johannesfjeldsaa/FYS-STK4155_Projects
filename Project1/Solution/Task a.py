import numpy as np

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


from Franke_function import FrankeFunction
from OLS import OLS
from setup import save_fig, data_path

if __name__ == '__main__':

    # Generate x, y meshgrid in order to implement the FrankeFunction

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)


    OLS_regression = OLS(5, x, y, z)

    OLS_regression.split_data(4/5)
    print(f'Split performed: {OLS_regression.splitted}')

    OLS_regression.standard_scaling()

    print(f'Scaling performed: {OLS_regression.scaled}\n'
          f'Scaling methode: {OLS_regression.scaling_methode}')


    OLS_regression.train_by_OLS(train_on_scaled=True)

    OLS_regression.predict_training()

    OLS_regression.predict_test()

    print(OLS_regression.y_test[0])
    print(OLS_regression.y_pred_test[0])
    print(OLS_regression.y_train_mean)


    # The mean squared error
    print(f'Mean squared error training: {OLS_regression.MSE(OLS_regression.y_train, OLS_regression.y_pred_train):.4f}')
    print(f'Mean squared error test: {OLS_regression.MSE(OLS_regression.y_test, OLS_regression.y_pred_test):.4f}')

    print(f'R^2 training: {OLS_regression.R_squared(OLS_regression.y_train, OLS_regression.y_pred_train):.4f}')
    print(f'R^2 squared error test: {OLS_regression.R_squared(OLS_regression.y_test, OLS_regression.y_pred_test):.4f}')

    z = FrankeFunction(x, y)
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


