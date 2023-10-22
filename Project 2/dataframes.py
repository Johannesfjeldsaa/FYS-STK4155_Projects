import pandas as pd
import numpy as np


"""
Class for making dataframes
"""

methods = np.array(['METHOD', 'no momentum', 'momentum', 'RMSprop', 'adagrad', 'adam'])
results_header = np.array(['batchsize MSE', 'epochs MSE', 'MSE score', 'batchsize R2', 'epochs R2', 'R2 score'])

df = pd.DataFrame(results_header, methods.T)
print(df)