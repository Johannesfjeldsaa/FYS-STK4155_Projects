import numpy as np

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


from Franke_function import FrankeFunction
from LinRegression import LinRegression
from setup import save_fig, data_path

if __name__ == '__main__':

    # Generate x, y meshgrid in order to implement the FrankeFunction

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')

    z = FrankeFunction(x_mesh, y_mesh)


    OLS_regression = LinRegression(5, x, y, z)
    print(np.shape(OLS_regression.X))
    print(np.shape(OLS_regression.y))

    OLS_regression.split_data(4/5)

    print(f'Split performed: {OLS_regression.splitted}')


    OLS_regression.scale()
    print(f'Scaling performed: {OLS_regression.scaled}\n'
          f'Scaling methode: {OLS_regression.scaling_method}')


    OLS_regression.train_model(train_on_scaled=True)
    print(f'The optimal parametres are: {OLS_regression.beta}')

    OLS_regression.predict_training()

    OLS_regression.predict_test()

    print(OLS_regression.y_test[0])
    print(OLS_regression.y_pred_test[0])
    print(OLS_regression.y_scaler)


    # The mean squared error
    print(f'Mean squared error training: {OLS_regression.MSE(OLS_regression.y_train, OLS_regression.y_pred_train):.4f}')
    print(f'Mean squared error test: {OLS_regression.MSE(OLS_regression.y_test, OLS_regression.y_pred_test):.4f}')

    print(f'R^2 training: {OLS_regression.R_squared(OLS_regression.y_train, OLS_regression.y_pred_train):.4f}')
    print(f'R^2 squared error test: {OLS_regression.R_squared(OLS_regression.y_test, OLS_regression.y_pred_test):.4f}')


    # Plot the surface.

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


