import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from . import nans
from . import encoder
from . import splitter
from . import cross_validation
from . import recover
from . import fit_pred_bagged
from . import feature_selection_bagged
from . import save_load_algorithms


# import datetime
# import pickle
# from os import makedirs

# from termcolor import colored
# from itertools import chain, combinations
# from multiprocessing import Pool, cpu_count

# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
# from seaborn import heatmap










def redefine_doc(docstr):
    """
    Decorator: adoption outer func.__doc__.
    """
    def _decorator(func):
        func.__doc__ = docstr
        return func
    return _decorator










class Snaplib:
    '''
    data preprocessing library
    '''
    ENCODER = encoder.Encoder()




    @staticmethod
    @redefine_doc(nans.nan_info.__doc__)
    def nan_info(df : pd.DataFrame) -> pd.DataFrame:
        return nans.nan_info(df)




    @staticmethod
    @redefine_doc(nans.nan_plot.__doc__)
    def nan_plot(df : pd.DataFrame) -> plt.figure:
        return nans.nan_plot(df)




    @staticmethod
    @redefine_doc(nans.cleane.__doc__)
    def cleane( df : pd.DataFrame, 
                target : str = None, 
                verbose : bool = True
                ) -> pd.DataFrame: 
        return nans.cleane(df, target, verbose)




    @redefine_doc(encoder.Encoder.encode_dataframe.__doc__)
    def encode_dataframe(self, 
                         df : pd.DataFrame, 
                         inplace : bool = False, 
                         ) -> pd.DataFrame: 
        if inplace is False:
            df_copied = df.copy()
            return self.ENCODER.encode_dataframe(df_copied)
        elif inplace is True:
            return self.ENCODER.encode_dataframe(df)



    
    @redefine_doc(encoder.Encoder.decode_dataframe.__doc__)
    def decode_dataframe(self, 
                         df : pd.DataFrame, 
                         inplace : bool = False, 
                         ) -> pd.DataFrame: 
        if inplace is False:
            df_copied = df.copy()
            return self.ENCODER.decode_dataframe(df_copied)
        elif inplace is True:
            return self.ENCODER.decode_dataframe(df)




    @staticmethod
    @redefine_doc(splitter.k_folds_split.__doc__)
    def k_folds_split(df : pd.DataFrame, 
                      target_name : str, 
                      k : int, 
                      ) -> dict: 
        return splitter.k_folds_split(df, target_name, k)




    @staticmethod
    @redefine_doc(splitter.train_test_split_balanced.__doc__)
    def train_test_split_balanced(  df : pd.DataFrame, 
                                    target_feature : str, 
                                    test_size : float = 0.2, 
                                    random_state: int = 0, 
                                    research : bool = False, 
                                    ) -> tuple:
        return  splitter.train_test_split_balanced( df, 
                                                    target_feature, 
                                                    test_size, 
                                                    random_state, 
                                                    research
                                                    )




    @staticmethod
    @redefine_doc(cross_validation.cross_val.__doc__)
    def cross_val(algorithms : list, 
                  k_fold_dict : dict, 
                  metric : Callable, 
                  task : str, 
                  cv : int, 
                  verbose : int or bool = 0, 
                  early_stopping_rounds : int = 0, 
                  ) -> float:
        return cross_validation.cross_val(
                                    algorithms, 
                                    k_fold_dict, 
                                    metric, 
                                    task, 
                                    cv, 
                                    verbose=verbose, 
                                    early_stopping_rounds=early_stopping_rounds
                                    )




    @staticmethod
    @redefine_doc(recover.recover_data.__doc__)
    def recover_data(df_in : pd.DataFrame, 
                     device : str = 'cpu',
                     verbose : int or bool = True,
                     discrete_columns : list or str = 'auto', 
                     ) -> pd.DataFrame:
        return recover.recover_data(df_in, 
                                    device=device,
                                    verbose=verbose,
                                    discrete_columns=discrete_columns, 
                                    )




    @staticmethod
    @redefine_doc(fit_pred_bagged.fit_stacked.__doc__)
    def fit_stacked(algorithms_list : list, 
                    X_train : pd.DataFrame, 
                    y_train : pd.Series or np.ndarray, 
                    X_val : pd.DataFrame = None, 
                    y_val : pd.Series or np.ndarray = None, 
                    verbose : int or bool = 0, 
                    early_stopping_rounds : int = 0, 
                    ) -> list:
        return  fit_pred_bagged.fit_stacked(
                    algorithms_list, 
                    X_train, 
                    y_train, 
                    X_val=X_val, 
                    y_val=y_val, 
                    verbose=verbose, 
                    early_stopping_rounds=early_stopping_rounds, 
                    )




    @staticmethod
    @redefine_doc(fit_pred_bagged.predict_stacked.__doc__)
    def predict_stacked(algorithms_list : list, 
                        X_pred : pd.DataFrame, 
                        task : str = 'clsf'
                        ) -> list:
        return  fit_pred_bagged.predict_stacked(algorithms_list, 
                                                X_pred, 
                                                task=task,
                                                )




    @staticmethod
    @redefine_doc(feature_selection_bagged.features_selection_regr.__doc__)
    def features_selection_regr(algorithms : list, 
                                df : pd.DataFrame, 
                                target : str, 
                                metric : Callable, 
                                cv : int, 
                                verbose : int or bool = 0, 
                                early_stopping_rounds : int = 0, 
                                ) -> list:
        return feature_selection_bagged.features_selection_regr(
                                    algorithms, 
                                    df, 
                                    target, 
                                    metric, 
                                    cv, 
                                    verbose=verbose, 
                                    early_stopping_rounds=early_stopping_rounds, 
                                    )




    @staticmethod
    @redefine_doc(feature_selection_bagged.features_selection_clsf.__doc__)
    def features_selection_clsf(algorithms : list, 
                                df : pd.DataFrame, 
                                target : str, 
                                metric : Callable, 
                                cv : int, 
                                verbose : int or bool = 0, 
                                early_stopping_rounds : int = 0, 
                                ) -> list:
        return feature_selection_bagged.features_selection_clsf(
                    algorithms, 
                    df, 
                    target, 
                    metric, 
                    cv, 
                    verbose=verbose, 
                    early_stopping_rounds=early_stopping_rounds, 
                    )




    @staticmethod
    @redefine_doc(save_load_algorithms.save_stack.__doc__)
    def save_stack( algorithms_list : list, 
                    directory : str = ''
                    ) -> None:
        return save_load_algorithms.save_stack(algorithms_list, 
                                               directory=directory)




    @staticmethod
    @redefine_doc(save_load_algorithms.load_stack.__doc__)
    def load_stack( names_list : list, 
                    directory : str = ''
                    ) -> None:
        return save_load_algorithms.load_stack( names_list, 
                                                directory=directory, 
                                                )




#EOF