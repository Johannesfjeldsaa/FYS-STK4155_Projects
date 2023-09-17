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

    OLS_regression = OLS(x, y, z)


    OLS_regression.split_data(4/5)
    print(f'Split performed: {OLS_regression.splitted}')

    OLS_regression.standard_scaling()

    print(f'Scaling performed: {OLS_regression.scaled}\n'
          f'Scaling methode: {OLS_regression.scaling_methode}')


    OLS_regression.train_by_OLS(train_on_scaled=True)

    OLS_regression.predict_traing()

    OLS_regression.predict_test()






