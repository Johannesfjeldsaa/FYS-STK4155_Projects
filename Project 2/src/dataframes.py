import pandas as pd
import numpy as np


"""
Class for making dataframes
"""

def df_analysis_method_is_index(dict, methods, headers):
    """

    :param dict: contain method with lists of results # important
    :param methods: contain the methods that you want to add the results to
    :param header: contains the results (have to be in correct order) that you want to obtain
    :return: dataframe with your results
    """
    analysis_df = pd.DataFrame(index=methods, columns=headers)

    for key, row in dict.items():  # key is the index, key contains list
        for value, header in zip(row, headers):
            analysis_df.loc[key, header] = value

    return analysis_df