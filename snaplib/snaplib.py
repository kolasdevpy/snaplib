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





class Snaplib:
    '''
    data preprocessing library
    '''
    ENCODER = encoder.Encoder()






    @staticmethod
    def nan_info(df : pd.DataFrame) -> pd.DataFrame:
        '''
        Get pandas.DataFrame with information about missing data in columns
        Use case:
        missing_info_df = Snaplib().nan_info(df)
        '''
        return nans.nan_info(df)




    @staticmethod
    def nan_plot(df : pd.DataFrame) -> plt.figure:
        '''
        Visualise missing data in pandas.DataFrame
        Use case:
        Snaplib().nan_plot(df)
        '''
        return nans.nan_plot(df)




    @staticmethod
    def cleane( df : pd.DataFrame, 
                target : str = None, 
                verbose : bool = True
                ) -> pd.DataFrame: 
        '''
        drop_duplicates
        drop rows with nan in target
        drop column with 1 unique value
        Use case:
        df = Snaplib().cleane(df, target_name, verbose=True)
        '''
        return nans.cleane(df, target, verbose)




    def encode_dataframe(self, 
                         df : pd.DataFrame, 
                         inplace : bool = False, 
                         ) -> pd.DataFrame: 
        '''
        encode a dataframe
        Use case:
        df = Snaplib().encode_dataframe(df)
        '''
        if inplace is False:
            df_copied = df.copy()

            return self.ENCODER.encode_dataframe(df_copied)
        elif inplace is True:
            return self.ENCODER.encode_dataframe(df)



        
    def decode_dataframe(self, 
                         df : pd.DataFrame, 
                         inplace : bool = False, 
                         ) -> pd.DataFrame: 
        '''
        encode a dataframe
        Use case:
        df = Snaplib().decode_dataframe(df)
        '''
        if inplace is False:
            df_copied = df.copy()
            return self.ENCODER.decode_dataframe(df_copied)
        elif inplace is True:
            return self.ENCODER.decode_dataframe(df)




    @staticmethod
    def k_folds_split(df : pd.DataFrame, 
                      target_name : str, 
                      k : int, 
                      ) -> dict: 
        '''
        Return a dictionary of lists of DataFrames and Series for target
        with next structure:

        k_fold_dict = { 
                        'train_X' : [train_df_0, train_df_1, ... , train_df_k],
                        'test_X'  : [etc.], 
                        'train_y' : [series_0, series_1, ... , series_k],
                        'test_y'  : [etc.],
                      }

        Use case:
        k_fold_dict_data = Snaplib().k_folds_split(df, target_name, k)
        '''
        return splitter.k_folds_split(df, target_name, k)




    @staticmethod
    def train_test_split_balanced(  df : pd.DataFrame, 
                                    target_feature : str, 
                                    test_size : float = 0.2, 
                                    random_state: int = 0, 
                                    research : bool = False, 
                                    ) -> tuple:
        ''' 
        Split the data with the distribution as close as possible 
        to the same in both the train set and the test set, and not only for the target column, 
        but also for all other columns.

        Use case:
        train_X, test_X, train_y, test_y = Snaplib().train_test_split_balanced(df, target_name, test_size=0.2, random_state=0, research=True)

        1) The input should be a whole DataFrame. It's the first positional argument.
        2) And the second positional argument should be the name of the target feature as a string.
        3) This method has internal testing.
        In this way you can testing the usefulness of the custom split.
        The first test performing with a random_state values from 0 to 1/test_size by sklearn.model_selection.train_test_split.
        Before split performing a final testing with Snaplib().train_test_split_balanced.

        You can perform testing by specifying the value of the research argument = True.

        4) If you are convinced that the method is useful. You can silence the method.
        Set the research argument to False.

        5) The number of possible random_state is an equivalent to 1/test_size.

        TESTS on https://www.kaggle.com/artyomkolas/train-test-split-balanced-custom-in-snaplib/notebook
        '''
        return  splitter.train_test_split_balanced( df, 
                                                    target_feature, 
                                                    test_size, 
                                                    random_state, 
                                                    research
                                                    )




    @staticmethod
    def cross_val(algorithms : list, 
                  k_fold_dict : dict, 
                  metric : Callable, 
                  task : str, 
                  cv : int, 
                  verbose : int or bool = 0, 
                  early_stopping_rounds : int = 0, 
                  ) -> float:
        ''' 
        Cross Validation method for list of algorithms.
        
        Use case:
        score = Snaplib().cross_val(algorithms, 
                                    k_fold_dict, 
                                    metric, 
                                    task, 
                                    cv, 
                                    verbose=0, 
                                    early_stopping_rounds=0)
        
        algorithms is a list of algorithms like algs = [
                                                        [LGBMClassifier, dict(params)],
                                                        [XGBClassifier, dict(params)], 
                                                        [CatBoostClassifier, dict(params)],
                                                        ]
        k_fold_dict is a dictionary with the structure:
        
        K_FOLD = 3
        k_fold_dict = { 
                        'train_X' : [df_0, df_1, df_2],
                        'test_X'  : [df_0, df_1, df_2],
                        'train_y' : [seri_0, seri_1, seri_2],
                        'test_y'  : [seri_0, seri_1, seri_2],
                       }

        Get k_fold_dict:
        k_fold_dict = Snaplib().k_folds_split(df, target_name_str, K_FOLD)
              
        metric is a metric like f1_score or mean_absolute_error : Callable.
        task='clsf' or 'regr', classification or regression.
        cv is num K_FOLD integer 
        verbose = 0 mute, 1 verbose.
        early_stopping_rounds default 0 or positive int.
        '''
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
    def recover_data(df_in : pd.DataFrame, 
                     device : str = 'cpu',
                     verbose : int or bool = True,
                     discrete_columns : list or str = 'auto', 
                     ) -> pd.DataFrame:
        ''' 
        Imputing of missing values (np.nan) in tabular data, not TimeSeries.

        Use case:
        df = Snaplib().recover_data(df, verbose=True, discrete_columns="auto")
        device must be "cpu" or "gpu". Sometime small datasets work faster with cpu.
        if set verbose=True algorithm runs cross validation tests and print results of tests for decision making.
        discrete_columns = ['col_name_1', 'col_name_2', 'col_name_3', 'etc']

        TESTS on https://www.kaggle.com/code/artyomkolas/nan-prediction-in-progress/notebook
        '''
        return recover.recover_data(df_in, 
                                    device=device,
                                    verbose=verbose,
                                    discrete_columns=discrete_columns, 
                                    )





    @staticmethod
    def fit_stacked(algorithms_list : list, 
                    X_train : pd.DataFrame, 
                    y_train : pd.Series or np.ndarray, 
                    X_val : pd.DataFrame = None, 
                    y_val : pd.Series or np.ndarray = None, 
                    verbose : int or bool = 0, 
                    early_stopping_rounds : int = 0, 
                    ) -> list:
        ''' 
        Fit method for list of algorithms.
        
        Use case:
        algorithms_list = Snaplib().fit_stacked(
                                                algorithms_list, 
                                                X_train, 
                                                y_train, 
                                                X_val=None, 
                                                y_val=None,
                                                verbose=0,
                                                ):
        
        algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        X_train and y_train are data for training list of algorithms.

        verbose = 0 mute, 1 verbose.
        early_stopping_rounds default 0 or positive int.
        '''
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
    def predict_stacked(algorithms_list : list, 
                        X_pred : pd.DataFrame, 
                        task : str = 'clsf'
                        ) -> list:
        ''' 
        Prediction method for list of algorithms.
        
        Use case:
        y_hat = predict_stacked(self, 
                                algorithms_list, 
                                X, 
                                task='clsf'
                                ):
        
        algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        X_pred is dataframe for prediction.
        task='clsf' or 'regr', classification or regression
        '''
        return  fit_pred_bagged.predict_stacked(algorithms_list, 
                                                X_pred, 
                                                task=task,
                                                )




    @staticmethod
    def features_selection_regr(algorithms : list, 
                                df : pd.DataFrame, 
                                target : str, 
                                metric : Callable, 
                                cv : int, 
                                verbose : int or bool = 0, 
                                early_stopping_rounds : int = 0, 
                                ) -> list:
        '''
        Select bests features for modeling and return list with bad features required to be droped.

        Use case:
        features_to_drop = Snaplib().features_selection_regr(algorithms, 
                                                             df, 
                                                             target, 
                                                             metric, 
                                                             cv, 
                                                             verbose=0, 
                                                             early_stopping_rounds=0)
        
        df.drop(features_to_drop, inplace=True, axis=1)


        algorithms is a list of algorithms like algs = [
                                                        [LGBMRegressor, dict(params)],
                                                        [XGBRegressor, dict(params)], 
                                                        [CatBoostRegressor, dict(params)],
                                                        ]

        df = pandas.core.frame.DataFrame.
        target = name of target of str type.

        metric is a metric like f1_score or mean_absolute_error : Callable.
        cv is num K_FOLD integer 
        verbose = 0 mute, 1 verbose.
        early_stopping_rounds default 0 or positive int.
        '''
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
    def features_selection_clsf(algorithms : list, 
                                df : pd.DataFrame, 
                                target : str, 
                                metric : Callable, 
                                cv : int, 
                                verbose : int or bool = 0, 
                                early_stopping_rounds : int = 0, 
                                ) -> list:
        ''' 
        Select bests features for modeling and return list with bad features required to be droped.
        
        Use case:
        features_to_drop = Snaplib().features_selection_clsf(algorithms, 
                                                             df, 
                                                             target, 
                                                             metric, 
                                                             cv, 
                                                             verbose=0,  
                                                             early_stopping_rounds=0)
        
        df.drop(features_to_drop, inplace=True, axis=1)


        algorithms is a list of algorithms like algs = [
                                                        [LGBMClassifier, dict(params)],
                                                        [XGBClassifier, dict(params)], 
                                                        [CatBoostClassifier, dict(params)],
                                                        ]

        df = pandas.core.frame.DataFrame.
        target = name of target of str type.
                
        metric is a metric like f1_score or mean_absolute_error : Callable.
        cv is num K_FOLD integer 
        verbose = 0 mute, 1 verbose.
        early_stopping_rounds default 0 or positive int.
        '''
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
    def save_stack( algorithms_list : list, 
                    directory : str = ''
                    ) -> None:
        ''' 
        Save method for all in list of algorithms in directory.
        Return list of file names
        
        Use case:
        file_names = save_stack(algorithms_list, directory='')
        
        algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        directory:
        '' - save files to current working directory
        or
        save files to /some/directory/not/exist/
        '''
        return save_load_algorithms.save_stack(algorithms_list, 
                                               directory=directory)




    @staticmethod
    def load_stack( names_list : list, 
                    directory : str = ''
                    ) -> None:
        ''' 
        Load method for file names in list of names in directory.
        Return list of algorithms.
        
        Use case:
        algorithms = load_stack(names_list, directory='')
        
        names_list is the list of names like ['LGBMClassifier.sav', 'XGBClassifier.sav', 'CatBoostClassifier.sav']
        directory:
        '' - read files from current working directory
        or
        read files from /some/directory/not/exist/
        '''
        return save_load_algorithms.load_stack( names_list, 
                                                directory=directory, 
                                                )




#EOF