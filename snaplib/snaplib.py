import datetime
import sys
import pickle
from os import makedirs
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from termcolor import colored
from itertools import chain, combinations
from multiprocessing import Pool, cpu_count


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix











class Snaplib:
    '''
    data preprocessing library
    '''
    
    def __init__(self, encoder_pool=dict(), decoder_pool=dict(), encoded_columns=list()):
        self.encoder_pool = encoder_pool
        self.decoder_pool = decoder_pool
        self.encoded_columns = encoded_columns
    

    
    
    @property
    def encoder_pool(self):
        return self.__encoder_pool

    @encoder_pool.setter
    def encoder_pool(self, encoder_pool):
        self.__encoder_pool = encoder_pool
    
    
    @property
    def decoder_pool(self):
        return self.__decoder_pool

    @decoder_pool.setter
    def decoder_pool(self, decoder_pool):
        self.__decoder_pool = decoder_pool
    
    
    @property
    def encoded_columns(self):
        return self.__encoded_columns

    @encoded_columns.setter
    def encoded_columns(self, encoded_columns):
        self.__encoded_columns = encoded_columns
        
        
    

    class Counter:
        def __init__(self):
            self.__counter = 0

        def __call__(self, step=1):
            self.__counter += step
            return self.__counter
    



    
        
        



    def nan_info(self, df):
        '''
        Get pandas.DataFrame with information about missing data in columns

        Use case:
        missing_info_df = Snaplib().nan_info(df)
        '''
        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')

        data_info = pd.DataFrame(index=df.columns)
        try:
            data_info['NaN_counts'] = df[[col for col in df.columns if df[col].isna().sum() > 0]].isna().sum().sort_values(ascending = True)
            data_info['NaN_percent'] = data_info['NaN_counts'].apply(lambda x: round((x/len(df))*100, 2))
            data_info['col_type'] = df.dtypes
            data_info = data_info.sort_values(by=['NaN_counts'], ascending=True)
        except:
            return data_info
        return data_info










    def nan_plot(self, df):
        '''
        Visualise missing data in pandas.DataFrame

        Use case:
        Snaplib().nan_plot(df)
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')

        plt.figure(figsize=(int(len(df.columns)/4) if len(df.columns)>30 else 10, 10))
        plt.pcolor(df.isnull(), cmap='Blues_r')
        plt.yticks([int(el*(len(df)/10)) for el in range(0, 10)])
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=80)
        plt.show()
        return plt









   
    def cleane(self, df, target=None, verbose=True):
        '''
        drop_duplicates, 
        drop rows with nan in target, 
        drop column with 1 unique value

        Use case:
        df = Snaplib().cleane(df, target_name, verbose=True)
        
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if target is not None:
            if type(target) != str:
                raise TypeError('target must be of str type.')
        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')
        
        # DROP DUPLICATES
        start_shape = df.shape
        if verbose:
            print(f'Start shape: {start_shape}\n\n')
            print(f'DROP DUPLICATES:')
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        dr_dupl_shape = df.shape
        if verbose:
            print(f'{start_shape[0] - dr_dupl_shape[0]} rows have been dropped')
            print(f'shape: {dr_dupl_shape}\n')

        # DROP COLUMNS with 1 unique value
            print('DROP COLUMNS with 1 unique value:')
            
            count_drop_columns = 0
        for col in df.columns:
            unique_array = np.array(df[col].unique())

            if len(unique_array) == 1:
                count_drop_columns += 1
                df.drop([col], inplace=True, axis=1)
                if verbose:
                    print(f'column "{col}" cnontains 1 unique value - has been dropped')
            elif len(unique_array) == 2 and np.any(pd.isnull(df[col])):
                if verbose:
                    print(f'!!! column "{col}" cnontains 1 unique value and np.nan')
        
        if verbose:
            print(f'{count_drop_columns} columns have been dropped')
            print(f'shape: {df.shape}\n')

        # DROP ROWS with NaN IN TARGET
        if target:
            if verbose:
                print('DROP ROWS with NaN IN TARGET:')
            nan = df[df[target].isnull()]
            indeces = list(nan.index)
            if verbose:
                print(f'{len(indeces)} rows have been dropped')
            df = df.drop(df.index[indeces])
            df.reset_index(drop=True, inplace=True)
        if verbose:
            print(f'shape: {df.shape}\n')
            print(f'\nFinish shape: {df.shape}\n')
        return df

    
    
    
    
    
    
    
    
    
    
    
    def encode_column(self, df, column):
        '''
        encode one column in dataframe

        Use case:
        df = Snaplib().encode_column(df, column_name_str)
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(column) != str:
            raise TypeError('column must be of str type.')


        self.encoder_pool[column] = {}
        self.decoder_pool[column] = {}
        not_nan_index=df[df[column].notnull()].index
        values_set = list(set(list(df.loc[not_nan_index, column])))   # must be sorted?
        value = 0.0
        for el in values_set:
            self.encoder_pool[column][el] = value
            self.decoder_pool[column][value] = el
            value += 1
        df[column] = df[column].map(self.encoder_pool[column])
        df[column] = df[column].astype('float64')
        return df
    
    
    
    
    
    
    
    
    
    def decode_column(self, df, column):
        '''
        decode one column in dataframe

        Use case:
        df = Snaplib().decode_column(df, column_name_str)
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(column) != str:
            raise TypeError('column must be of str type.')

        df[column] = df[column].map(self.decoder_pool[column])
        df[column] = df[column].astype('object')
        return df
    
    
    
    
    
    
    



    def encode_dataframe(self, df_0):
        '''
        encode a dataframe

        Use case:
        df = Snaplib().encode_dataframe(df)
        '''

        if not isinstance(df_0, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')

        df = df_0.copy()

        self.encoder_pool = {}
        self.decoder_pool = {}
        self.encoded_columns = []

        types = df.dtypes
        for col in df.columns:
            feature_type = types[col]
            if feature_type == 'object' or feature_type == 'bool':            
                df[col] = self.encode_column(df[[col]], col)
                self.encoded_columns.append(col)
            else:
                # object type column with NaN sometimes has type float64
                try:
                    df.loc[:, col] = df.loc[:, col] + 0
                except:
                    df[col] = df[col].astype('object')
                    df[col] = self.encode_column(df[[col]], col)
                    self.encoded_columns.append(col)
        return df
    
    
    
    
    
    
    
    
    

    
    def decode_dataframe(self, df):
        '''
        decode a dataframe

        Use case:
        df = Snaplib().decode_dataframe(df)
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')


        for col in self.encoded_columns:
            df[col] = self.decode_column(df[[col]], col)
        return df


    
    
    
    
    
    
    
    
    
    def k_folds_split(self, df, target_name, k):

        '''
        Return a dictionary list of DataFrames and Series for target
        with next structure:

        k_fold_dict = { 
                        'train_X' : [train_df_0, train_df_1, ... , train_df_k],
                        'test_X'  : [etc.], 
                        'train_y' : [series_0, series_1, ... , series_k],
                        'test_y'  : [etc.],
                      }

        Use case:
        k_fold_dict_data = Snaplib().k_folds_split( df, target_name, k)
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(target_name) != str:
            raise TypeError('target_name must be of str type.')
        if type(k) != int:
            raise TypeError('k must be of int type.')

        k_fold_dict = { 
                        'train_X' : [],
                        'test_X'  : [], 
                        'train_y' : [],
                        'test_y'  : [],
                      }


        for i in range(0, k):
            train_X, test_X, train_y, test_y = self.train_test_split_balanced(df, 
                                                                              target_name, 
                                                                              random_state=i, 
                                                                              test_size=1/k, 
                                                                              research=False
                                                                              )
            k_fold_dict['train_X'].append(train_X)
            k_fold_dict['test_X'].append(test_X)
            k_fold_dict['train_y'].append(train_y)
            k_fold_dict['test_y'].append(test_y)

        return k_fold_dict


    
    
    
    
    
    
    
    
    
    
    
    def fit_predict_stacked(self, 
                            algorithms_list, 
                            X_train, 
                            y_train, 
                            X_test, 
                            y_test=None, 
                            task='clsf', 
                            verbose=0
                            ):
        
        ''' 
        Prediction method for list of algorithms.
        
        Use case:
        y_hat = Snaplib().fit_predict_stacked(
                                        algorithms_list, 
                                        X_train, 
                                        y_train, 
                                        X_test, 
                                        y_test or None, 
                                        task='clsf' or 'regr', 
                                        verbose= 0, 1, 2
                                        ):
        
        algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        X_train and y_train are data for training list of algorithms.
        X_test is dataframe for prediction.
        y_test optionaly. If exist visualize it as last column on a plot (verbose=True). 
        task='clsf' or 'regr', classification or regression
        verbose = 0 mute, 1 verbose.

        '''
        if type(algorithms_list) != list:
            raise TypeError('algorithms_list must be of list type.')
        if not isinstance(X_train, pd.core.frame.DataFrame):
            raise TypeError('The X__train must be a pandas.core.frame.DataFrame instance.')
        if not isinstance(X_test, pd.core.frame.DataFrame):
            raise TypeError('The X__pred must be a pandas.core.frame.DataFrame instance.')

        if not isinstance(y_train, pd.core.frame.Series) and not isinstance(y_train, np.ndarray):
            raise TypeError('The y__train must be a pandas.core.frame.Series instance.')
        if y_test is not None:
            if not isinstance(y_test, pd.core.frame.Series) and not isinstance(y_test, np.ndarray):
                raise TypeError('The y__test must be a pandas.core.frame.Series instance or numpy.ndarray.')

        if task !='clsf' and task != 'regr':
            raise ValueError('Task in fit_predict_stacked() must be "clsf" or "regr".')
        if len(algorithms_list) == 0:
            raise ValueError('Algorithms list is empty.')

        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')


        stacked_predicts = pd.DataFrame()
        alg_names = []
        increment = 1
        for algorithm in algorithms_list:
            alg = algorithm[0]
            params = algorithm[1]
            alg_name = alg.__name__[:3]
            if alg_name in alg_names:
                alg_name = alg_name + '_' + str(increment)
                increment+=1
            if y_test is not None and alg_name in ['LGB', 'XGB', 'Cat']:
                model = alg(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=200, verbose=False)
            else:
                model = alg(**params)
                model.fit(X_train, y_train)

            y_hat = model.predict(X_test)
            if task =='clsf':
                stacked_predicts[alg_name] = y_hat.astype('int64')
            elif task=='regr':
                stacked_predicts[alg_name] = y_hat
            alg_names.append(alg_name)

        if task =='clsf':
            stacked_predicts['Y_HAT_STACKED'] = stacked_predicts[alg_names].mode(axis=1)[0].astype('int64')
            if y_test is not None:
                stacked_predicts['Y_TEST'] = y_test.values.astype('int64')
        elif task=='regr':
            stacked_predicts['Y_HAT_STACKED'] = stacked_predicts[alg_names].mean(axis=1)
            if y_test is not None:
                stacked_predicts['Y_TEST'] = y_test.values

        return stacked_predicts


    
    
    
    
    
        
    

    def cross_val(self, algorithms, k_fold_dict, metric, task, cv, verbose=0):

                
        ''' 
        Cross Validation method for list of algorithms.
        
        Use case:
        score = Snaplib().cross_val(algorithms, k_fold_dict, metric, task, cv, verbose=0):
        
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
              
        metric is a metric like f1_score or mean_absolute_error.
        task='clsf' or 'regr', classification or regression.
        cv is num K_FOLD integer 
        verbose = 0 mute, 1 verbose.

        '''

        if type(algorithms) != list:
            raise TypeError('The algorithms must be of list type.')
        if len(algorithms) == 0:
            raise ValueError('algorithms_listt is empty.')
        if type(k_fold_dict) != dict:
            raise TypeError('The k_fold_dict must be of dict type.')
        if task != 'clsf' and task != 'regr':
            raise ValueError('task must be "clsf" or "regr".')
        if type(cv) != int:
            raise TypeError('cv must be of int type.')
        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')


        global num_combinations
        global all_prediction_df
        global counter
        global cpu_num
        
        
        results=[]    
        all_prediction_df = pd.DataFrame()
        counter = self.Counter()
        research_best_score=dict()
        cpu_num = cpu_count()
        num_combinations = 0
        
        
        def powerset(names_ls):
            return chain.from_iterable(combinations(names_ls, r) for r in range(len(names_ls)+1))


        
        
        for k in range(0, cv):
            pred_frame = self.fit_predict_stacked(algorithms, 
                                                    k_fold_dict['train_X'][k], 
                                                    k_fold_dict['train_y'][k], 
                                                    k_fold_dict['test_X'][k], 
                                                    k_fold_dict['test_y'][k], 
                                                    task,
                                                    verbose,
                                                    )

            pred = pred_frame.loc[:, 'Y_HAT_STACKED']
            score = metric(k_fold_dict['test_y'][k], pred)
            results.append(score)

            if verbose:
                all_prediction_df = pd.concat([all_prediction_df, pred_frame], ignore_index=True)

        results_std = np.std(results)
        score = sum(results) / len(results)


        if verbose:
            all_prediction_df.reset_index(drop=True, inplace=True)
            for alg in algorithms:
                print(alg[0].__name__)

            print('')
            if str(metric).split('.')[0] == 'functools':
                metric_name = str(metric).split('function ')[1].split(' ')[0]
            else:
                metric_name = metric.__name__
            print("%s %0.6f (std: +/- %0.2f)" % (metric_name, score, results_std))
            print('\n', results, '\n')


            y_test = all_prediction_df.loc[:, 'Y_TEST']
            y_pred = all_prediction_df.loc[:, 'Y_HAT_STACKED']
            if task == 'clsf':
                print('\n', classification_report(y_test, y_pred), '\n')
                plt.figure(figsize=(3, 3))
                cm = confusion_matrix(y_test, y_pred)
                heatmap(cm, annot=True, cmap="Blues", fmt='.0f',  cbar=False)
                plt.show()
            elif task == 'regr':
                print('')
                print('MAE    : %0.6f ' % mean_absolute_error(y_test, y_pred))
                print('RMSE   : %0.6f ' % mean_squared_error(y_test, y_pred)**0.5)
                print('R2     : %0.6f ' % r2_score(y_test, y_pred))


            # SHOW PLOTS
            rng = np.random.default_rng()
            idx = set(rng.integers(len(all_prediction_df)-1, size=200))

            data = all_prediction_df.loc[idx,:]
            data.sort_values(by='Y_TEST', ascending=True, inplace=True)
            data.reset_index(drop=True, inplace=True)
            if task =='clsf':
                plt.figure(figsize=(5, 10))
                plt.pcolor(data, cmap='Blues_r')
                cols = list(data.columns)
                plt.xticks(np.arange(0.5, len(cols), 1), cols, rotation=80)
                plt.show()

            elif task=='regr':
                plt.figure(figsize=(20, 10))
                for col in data.columns:
                    width = 5 if col == 'Y_TEST' else 1
                    plt.plot(data[col], label=col, linewidth=width)
                plt.legend()
                plt.xticks([])
                plt.title('RANDOM 200 cases sorted by test value')
                plt.show()

                
            # BEST COMBINATION RESEARCH
            all_prediction_df_columns = list(all_prediction_df.columns)
            if len(all_prediction_df_columns) > 3:
                print('\nThe Best Algorithms Combination:')
                print('Quadratic Complexity O(num_algorithms^2)')
                alg_names = all_prediction_df_columns[:]
                alg_names.remove('Y_TEST')
                alg_names.remove('Y_HAT_STACKED')

                all_combinations = list(powerset(alg_names))
                num_combinations = len(all_combinations) -1

                list_metrics = [metric]*num_combinations
                params = zip(all_combinations[1:], list_metrics)

                results_pool = []
                if task =='clsf':
                    with Pool(cpu_num) as p:
                        results_pool.append(p.map(self.get_score_clsf, params))
                elif task=='regr':
                    with Pool(cpu_num) as p:
                        results_pool.append(p.map(self.get_score_regr, params))

                for line in results_pool[0]:
                    key, value = line[0], line[1]
                    research_best_score[key] = value
                    
                print(f'\r{num_combinations} / {num_combinations} processed', end='', flush=True)
                



                best_list=[]
                ascending = True if task=='clsf' else False
                research_best_scoredict = sorted(research_best_score.items(), reverse=ascending)
                print(f'\n{metric_name}:          combination\n')
                i=0
                color='grey'
                for k, v in research_best_scoredict:
                    ks, vs  = str(k), str(v)
                    if i == 0:
                        color='green'
                        best_list = v[:]
                    if vs == str(alg_names):
                        color='blue'
                        vs = vs + ' - full stack of algorithms'
                    print("{:30s} {:100s} ".format(colored(ks, color), colored(vs, color)))
                    color='grey'
                    i+=1

                    
                # THE  BEST  SHOW  PLOTS
                if task =='clsf':
                    best_df = data[list(best_list)]
                    best_df['Y_HAT_STACKED'] = data.loc[:, list(best_list)].mode(axis=1)[0].astype('int64')
                    best_df['Y_TEST'] = data['Y_TEST']

                    f, ax = plt.subplots(1, 2, figsize=(10, 10))

                    plt.subplot(1, 2, 1)
                    plt.pcolor(data, cmap='Blues_r')
                    cols = list(all_prediction_df_columns)
                    plt.xticks(np.arange(0.5, len(cols), 1), cols, rotation=80)
                    ax[0].set_title('ALL ALGORITHMS')

                    plt.subplot(1, 2, 2)
                    plt.pcolor(best_df, cmap='Blues_r')
                    cols = list(best_list)+['Y_HAT_STACKED']+['Y_TEST']
                    plt.xticks(np.arange(0.5, len(cols), 1), cols, rotation=80)
                    ax[1].set_title('BEST COMPOSITION')

                    f.tight_layout()
                    
                    plt.show()
                    



                elif task=='regr':
                    plt.figure(figsize=(20, 10))
                    y_hat_all = data.loc[:, list(alg_names)].mean(axis=1)
                    y_hat_best = data.loc[:, list(best_list)].mean(axis=1)

                    plt.plot(y_hat_best, label=str(best_list), linewidth=1, color='green')
                    plt.plot(y_hat_all, label=str(alg_names), linewidth=1, color='blue')
                    plt.plot(data['Y_TEST'], label='Y_TEST', linewidth=5, color='red')
                    plt.legend()
                    plt.xticks([])
                    plt.title('Compare all algorithms and the best')
                    plt.show()
                    
        del num_combinations
        del all_prediction_df
        del counter
        del cpu_num

        return score
        
    @staticmethod
    def get_score_clsf(args):
        print(f'\r{counter(cpu_num)} / {num_combinations} processed', end='', flush=True)
        algs_list = list(args[0])
        metric = args[1]
        y_hat_set = all_prediction_df.loc[:, algs_list].mode(axis=1)[0].astype('int64')
        score_set = metric(all_prediction_df.loc[:, ['Y_TEST']], y_hat_set)
        return score_set, algs_list
    
    @staticmethod
    def get_score_regr(args):
        print(f'\r{counter(cpu_num)} / {num_combinations} processed', end='', flush=True)
        algs_list = list(args[0])
        metric = args[1]
        y_hat_set = all_prediction_df.loc[:, algs_list].mean(axis=1)
        score_set = metric(all_prediction_df.loc[:, ['Y_TEST']], y_hat_set)
        return score_set, algs_list










    def fit_stacked(self, 
                    algorithms_list, 
                    X_train, 
                    y_train, 
                    # X_val=None, 
                    # y_val=None, 
                    verbose=0
                    ):
        
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
        '''

        if type(algorithms_list) != list:
            raise TypeError('algorithms_list must be of list type.')
        if len(algorithms_list) == 0:
            raise ValueError('algorithms_list is empty.')
        
        if not isinstance(X_train, pd.core.frame.DataFrame):
            raise TypeError('The X__train must be a pandas.core.frame.DataFrame instance.')
        if not isinstance(y_train, pd.core.frame.Series) and not isinstance(y_train, np.ndarray):
            raise TypeError('The y__train must be a pandas.core.frame.Series instance or numpy.ndarray.')

        # if X_val is not None:
        #     if not isinstance(X_val, pd.core.frame.DataFrame):
        #         raise TypeError('The X__val must be a pandas.core.frame.DataFrame instance.')
        # if y_val is not None:
        #     if not isinstance(y_val, pd.core.frame.Series) and not isinstance(y_val, np.ndarray):
        #         raise TypeError('The y__val must be a pandas.core.frame.Series instance or numpy.ndarray.')

        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')


        for alg in algorithms_list:
            # alg_name = alg.__class__.__name__[:3]
            # if X_val is not None and y_val is not None and alg_name in ['LGB', 'XGB', 'Cat']:
            #     alg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=verbose)
            # else:
            alg.fit(X_train, y_train)

        return algorithms_list
    
    









    def predict_stacked(self, 
                        algorithms_list, 
                        X_pred, 
                        task='clsf'
                        ):

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
        if type(algorithms_list) != list:
            raise TypeError('algorithms_list must be of list type.')
        if len(algorithms_list) == 0:
            raise ValueError('algorithms_list is empty.')

        if not isinstance(X_pred, pd.core.frame.DataFrame):
            raise TypeError('The X__pred must be a pandas.core.frame.DataFrame instance.')

        if task !='clsf' and task != 'regr':
            raise ValueError('Task in fit_predict_stacked() must be "clsf" or "regr".')

        
        stacked_predicts = pd.DataFrame()
        alg_names = []
        for alg in algorithms_list:
            alg_name = alg.__class__.__name__[:3]
            if alg_name in alg_names:
                alg_name = alg_name + '_another'
            y_hat = alg.predict(X_pred)
            if task =='clsf':
                stacked_predicts[alg_name] = y_hat.astype('int64')
            elif task=='regr':
                stacked_predicts[alg_name] = y_hat
            alg_names.append(alg_name)

        if task =='clsf':
            stacked_predicts['Y_HAT_STACKED'] = stacked_predicts[alg_names].mode(axis=1)[0].astype('int64')
        elif task=='regr':
            stacked_predicts['Y_HAT_STACKED'] = stacked_predicts[alg_names].mean(axis=1)
        return stacked_predicts.loc[:, 'Y_HAT_STACKED']









    def features_selection_regr(self, algorithms, df, target, metric, cv, verbose=0):


        ''' 
        Select bests features for modeling and return list with bad features required to be droped.

        Use case:
        features_to_drop = Snaplib().features_selection_regr(algorithms, df, target, metric, cv, verbose=0):
        df.drop(features_to_drop, inplace=True, axis=1)


        algorithms is a list of algorithms like algs = [
                                                        [LGBMClassifier, dict(params)],
                                                        [XGBClassifier, dict(params)], 
                                                        [CatBoostClassifier, dict(params)],
                                                        ]

        df = pandas.core.frame.DataFrame.
        target = name of target of str type.

        metric is a metric like f1_score or mean_absolute_error.
        cv is num K_FOLD integer 
        verbose = 0 mute, 1 verbose.

        '''

        if type(algorithms) != list:
            raise TypeError('The algorithms must be of list type.')
        if len(algorithms) == 0:
            raise ValueError('algorithms_listt is empty.')
        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(target) != str:
            raise TypeError('target_feature must be of str type.')
        if type(cv) != int:
            raise TypeError('cv must be of int type.')
        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')


        cv_score_of_bad_feature = 0
        droped_features = []
        features = list(df.columns)
        scores_df = pd.DataFrame(index = features, columns=['cv_score'])

        k_folds_dict_data = self.k_folds_split(df[features], target, cv)
        score_with_all_features  = self.cross_val(algorithms,
                                                k_folds_dict_data, 
                                                metric, 
                                                task='regr', 
                                                cv=cv, 
                                                verbose=0)
        if verbose:
            print(f'{score_with_all_features}     General cv_score with all features')


        while score_with_all_features >= cv_score_of_bad_feature:
            scores_df = pd.DataFrame(index = features, columns=['cv_score'])
            k_folds_dict_data = self.k_folds_split(df[features], target, cv)
            score_with_all_features  = self.cross_val(algorithms, 
                                                    k_folds_dict_data, 
                                                    metric, 
                                                    task='regr', 
                                                    cv=cv, 
                                                    verbose=0)
            if verbose:
                print('\n\n')
                print(f'{len(features)} number of features')
                print("{:1.8f}   {:20}  ".format(score_with_all_features, 'BASE cv_score with all features'))
                print('\n\nwithou feature\n')

            for without_feature in features:
                if without_feature != target:
                    fit_faetures = features[:]
                    fit_faetures.remove(without_feature)
                    k_folds_dict_data = self.k_folds_split(df[fit_faetures], target, cv)
                    score  = self.cross_val(algorithms, 
                                            k_folds_dict_data, 
                                            metric, 
                                            task='regr', 
                                            cv=cv, 
                                            verbose=0)
                    scores_df.loc[without_feature] = score
                    if verbose:
                        print("{:1.8f}   {:20}  ".format(score, without_feature))

            scores_df = scores_df.sort_values(by=['cv_score'], ascending=True)
            bad_feature = scores_df.index[0]
            cv_score_of_bad_feature = scores_df.iloc[0][0]

            if score_with_all_features >= cv_score_of_bad_feature:
                features.remove(bad_feature)
                droped_features.append(bad_feature)
                if verbose:
                    print('='*50,'\n')
                    print(f'    APPEND TO DROP    {bad_feature}')
                    print('='*50,'\n')
            else:
                if verbose:
                    print('\n\n')
                    print(f'These features have been droped:\n{droped_features}')
                    print('\n\n')
                return droped_features










    def features_selection_clsf(self, algorithms, df, target, metric, cv, verbose=0):
        

        ''' 
        Select bests features for modeling and return list with bad features required to be droped.
        
        Use case:
        features_to_drop = Snaplib().features_selection_clsf(algorithms, df, target, metric, cv, verbose=0):
        df.drop(features_to_drop, inplace=True, axis=1)


        algorithms is a list of algorithms like algs = [
                                                        [LGBMClassifier, dict(params)],
                                                        [XGBClassifier, dict(params)], 
                                                        [CatBoostClassifier, dict(params)],
                                                        ]

        df = pandas.core.frame.DataFrame.
        target = name of target of str type.
              
        metric is a metric like f1_score or mean_absolute_error.
        cv is num K_FOLD integer 
        verbose = 0 mute, 1 verbose.

        '''

        if type(algorithms) != list:
            raise TypeError('The algorithms must be of list type.')
        if len(algorithms) == 0:
            raise ValueError('algorithms_listt is empty.')
        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(target) != str:
            raise TypeError('target_feature must be of str type.')
        if type(cv) != int:
            raise TypeError('cv must be of int type.')
        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')


        cv_score_of_bad_feature = 1
        droped_features = []
        features = list(df.columns)
        scores_df = pd.DataFrame(index = features, columns=['cv_score'])
        
        k_folds_dict_data = self.k_folds_split(df[features], target, cv)
        score_with_all_features  = self.cross_val(algorithms,
                                                  k_folds_dict_data, 
                                                  metric, 
                                                  task='clsf', 
                                                  cv=cv, 
                                                  verbose=0)
        if verbose:
            print(f'{score_with_all_features}     General cv_score with all features')
        
        
        while score_with_all_features <= cv_score_of_bad_feature:
            scores_df = pd.DataFrame(index = features, columns=['cv_score'])
            k_folds_dict_data = self.k_folds_split(df[features], target, cv)
            score_with_all_features  = self.cross_val(algorithms, 
                                                      k_folds_dict_data, 
                                                      metric, 
                                                      task='clsf', 
                                                      cv=cv, 
                                                      verbose=0)
            if verbose:
                print('\n\n')
                print(f'{len(features)} number of features')
                print("{:1.8f}   {:20}  ".format(score_with_all_features, 'BASE cv_score with all features'))
                print('\n\nwithou feature\n')

            for without_feature in features:
                if without_feature != target:
                    fit_faetures = features[:]
                    fit_faetures.remove(without_feature)
                    k_folds_dict_data = self.k_folds_split(df[fit_faetures], target, cv)
                    score  = self.cross_val(algorithms, 
                                            k_folds_dict_data, 
                                            metric, 
                                            task='clsf', 
                                            cv=cv, 
                                            verbose=0)
                    scores_df.loc[without_feature] = score
                    if verbose:
                        print("{:1.8f}   {:20}  ".format(score, without_feature))

            scores_df = scores_df.sort_values(by=['cv_score'], ascending=False)
            bad_feature = scores_df.index[0]
            cv_score_of_bad_feature = scores_df.iloc[0][0]
            
            if score_with_all_features <= cv_score_of_bad_feature:
                features.remove(bad_feature)
                droped_features.append(bad_feature)
                if verbose:
                    print('='*50,'\n')
                    print(f'    APPEND TO DROP    {bad_feature}')
                    print('='*50,'\n')
            else:
                if verbose:
                    print('\n\n')
                    print(f'These features have been droped:\n{droped_features}')
                    print('\n\n')
                return droped_features        
        
        







    
    def save_stack(self, algorithms_list, directory=''):
        ''' 
        Save method for all in list of algorithms in directory.
        Return list of file names
        
        Use case:
        file_names = save_stack(self, algorithms_list, directory='')
        
        algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        directory:
        '' - save files to current working directory
        or
        save files to /some/directory/not/exist/
        '''

        if type(algorithms_list) != list:
            raise TypeError('algorithms_list must be of list type.')
        if len(algorithms_list) == 0:
            raise ValueError('algorithms_list is empty.')
        if type(directory) != str:
            raise ValueError('directory must be of str type.')

        names = []
        if directory:
            directory = (directory[:-1] if directory[-1] == '/' else directory) + '/'
        makedirs(directory, exist_ok=True)
        
        for alg in algorithms_list:
            filename = alg.__class__.__name__ + '.sav'
            if filename in names:
                filename =  alg.__class__.__name__  + '_another' + '.sav'
            path = directory + filename
            pickle.dump(alg, open(path, 'wb'))
            names.append(filename)
        return names










    def load_stack(self, names_list, directory=''):
        ''' 
        Load method for file names in list of names in directory.
        Return list of algorithms.
        
        Use case:
        algorithms = load_stack(self, names_list, directory='')
        
        names_list is the list of names like ['LGBMClassifier.sav', 'XGBClassifier.sav', 'CatBoostClassifier.sav']
        directory:
        '' - read files from current working directory
        or
        read files from /some/directory/not/exist/
        '''

        if type(names_list) != list:
            raise TypeError('names_list must be of list type.')
        if len(names_list) == 0:
            raise ValueError('algorithms_list is empty.')
        if type(directory) != str:
            raise ValueError('directory must be of str type.')


        algorithms_list = []
    
        if directory:
            directory = (directory[:-1] if directory[-1] == '/' else directory) + '/'
        
        for alg_name in names_list:
            algorithms_list.append(pickle.load(open(directory + alg_name, 'rb')))
        return algorithms_list
    
    
    
    
    





    def train_test_split_balanced(self,
                                df, 
                                target_feature, 
                                test_size=0.2, 
                                random_state=0, 
                                research=False
                                ):
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

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(target_feature) != str:
            raise TypeError('target_feature must be of str type.')
        if type(test_size) != float:
            raise TypeError('test_size must be of float type in [0.0, 1.0].')
        if type(random_state) != int:
            raise TypeError('random_state must be of int type.')
        if type(research) != bool:
            raise TypeError('research must be of bool type.')


        CLASSIFIER_FOR_UNIQUE_VALUES_LESS_THAN = 20
        df_count = pd.DataFrame()


        def get_predictors(df, target_feature):
            predictors = list(df.columns)
            predictors.remove(target_feature)
            return predictors


        def get_X(df, predictors):
            return df[predictors]


        def get_y(df, target_feature):
            return df[[target_feature]]


        def regression_score(train_X, test_X, train_y, test_y):
            model = lgb.LGBMRegressor(random_state=0).fit(train_X, train_y)
            predict = model.predict(test_X)
            return mean_absolute_error(predict, test_y)


        def classification_score(train_X, test_X, train_y, test_y):
            model = lgb.LGBMClassifier(random_state=0).fit(train_X, train_y.values.ravel())
            predict = model.predict(test_X)
            return f1_score(predict, test_y.values.ravel(), average='macro')


        def get_research(X, y, test_size, split):
            nums_research = int(1/test_size)
            results = []
            if len(y.value_counts()) > CLASSIFIER_FOR_UNIQUE_VALUES_LESS_THAN:
                metric_name = 'MAE'
                print('regression test')
                for rs in range(0, nums_research):
                    if split == 'sklearn':
                        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=rs)
                    if split == 'balanced':
                        train_X, test_X, train_y, test_y = train_test_split_ordered(X, y, random_state=rs, test_size=test_size)
                    score = regression_score(train_X, test_X, train_y, test_y)
                    results.append(score)
                print(f'{metric_name} with random_state from 0 to {nums_research - 1}:')
            else:
                metric_name = 'f1_score'
                print('classification test')
                if split == 'sklearn':
                    print('with stratify=y')
                for rs in range(0, nums_research):
                    if split == 'sklearn':
                        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
                    if split == 'balanced':
                        train_X, test_X, train_y, test_y = train_test_split_ordered(X, y, random_state=rs, test_size=test_size)
                    score = classification_score(train_X, test_X, train_y, test_y)
                    results.append(score)
                print(f'{metric_name}(average="macro") with random_state from 0 to {nums_research -1}:')
            results_std = np.std(results)
            score = sum(results) / len(results)
            print("%s %0.6f (std: +/- %0.4f)" % (metric_name, score, results_std))
            print(results, '\n')
            return 0


        def order_and_sort_table(df, important_functions):
            df = df.sort_values(by=important_functions, ascending=True)
            df = df.reset_index(drop=True)
            return df


        def train_test_split_ordered(X, y, random_state, test_size):
            train_indexes = []
            test_indexes = []
            len_idxs = X.shape[0]

            every_n_el = int(1/test_size)
            increment = random_state % len_idxs % every_n_el

            for el in X.index:
                if (el + increment) % every_n_el:
                    train_indexes.append(el)
                else:
                    test_indexes.append(el)

            train_X = X.iloc[train_indexes]
            train_y = y.iloc[train_indexes, 0]
            test_X = X.iloc[test_indexes]
            test_y = y.iloc[test_indexes, 0]

            return train_X, test_X, train_y, test_y


        def visualize(train_array, test_array, column_name, train_str, test_str):   
            n_bins = 20    
            f, ax = plt.subplots(1, 2, figsize=(13, 4))

            plt.subplot(1, 2, 1)
            plt.hist(train_array, bins=n_bins)
            plt.xlabel('values')
            plt.ylabel('counts')
            ax[0].set_title(column_name + train_str)

            plt.subplot(1, 2, 2)
            plt.hist(test_array, bins=n_bins)
            plt.xlabel('values')
            plt.ylabel('counts')
            ax[1].set_title(column_name + test_str)

            f.tight_layout()
            return None




        # main
        predictors = get_predictors(df, target_feature) 
        for el in df.columns:
            count = len(df[el].value_counts())
            df_count.loc[el, 'counts'] = count
        df_count.sort_values(by='counts', ascending=True, inplace=True)

        if research:
            print('TEST by sklearn.model_selection.train_test_split:')
            get_research(get_X(df, predictors), get_y(df, target_feature), test_size, split='sklearn')
            
            print('='*50,'\n')
            print('The Table has been ordered and sorted by columns:')
            print(df_count)
            print('')
            init_time = datetime.datetime.now()

        ordered_predictors = list(df_count.index)
        if len(df[target_feature].value_counts()) <= CLASSIFIER_FOR_UNIQUE_VALUES_LESS_THAN:
            ordered_predictors.remove(target_feature)
            ordered_columns = [target_feature] + ordered_predictors
        else:
            ordered_columns = ordered_predictors[:]
        df = df[ordered_columns]

        df = order_and_sort_table(df, ordered_columns)
        predictors = get_predictors(df, target_feature)


        train_X, test_X, train_y, test_y = \
        train_test_split_ordered(get_X(df, predictors), get_y(df, target_feature), random_state=random_state, test_size=test_size)
        



        if research:
            finish_time = datetime.datetime.now()
            requared_time = finish_time - init_time
            print(f'Split required time:  {str(requared_time)}\n')
            
            print('='*50,'\n')
            print('TEST by train_test_split_balanced:')
            get_research(get_X(df, predictors), get_y(df, target_feature), test_size, split='balanced')
            
            print('\n===============   DISTRIBUTIONS   ===============\n\n')
            visualize(train_y, test_y, target_feature, ' train_y_TARGET', ' test_y_TARGET')
            for column in train_X.columns:
                visualize(train_X[column], test_X[column], column, ' train_X', ' test_X')

        return train_X, test_X, train_y, test_y










    def recover_data(self,
                     df_0, 
                     device='cpu',
                     verbose = 1,
                     discrete_columns='auto', 
                     ):
        ''' 
        Imputing of missing values (np.nan) in tabular data, not TimeSeries.

        Use case:
        df = Snaplib().recover_data(df, verbose=True, stacking=True)
        device must be "cpu" or "gpu". Sometime small datasets work faster with cpu.
        if set verbose = if True algorithm runs cross validation tests and print results of tests for decision making.
        discrete_columns = ['col_name_1', 'col_name_2', 'col_name_3', 'etc']
        '''

        if not isinstance(df_0, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if device !='cpu' and device != 'gpu':
            raise ValueError('device must be "cpu" or "gpu".')
        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')
        if not isinstance(discrete_columns, list) and discrete_columns != 'auto':
            raise TypeError('The discrete_columns must be a list instance or string "auto".')
        


        counter_predicted_values = 0
        CLASS_VALUE_COUNTS = 30
        K_FOLDS = 4
        
        
        warning_switch = 'WARNING:\nYou have a lot of unique values for this discrete column, more tahn 100.\
                                    \nPrediction and imputing has been switched to regression. \
                                    \nTry encode this column and restart this algorithm.\
                                    \nOtherwise you can include this feature to argument discrete_columns=[] \
                                    \nfor forced classification. It will take a very long time.\n'

        warning_time = 'WARNING:\nYou have a lot of unique values for this column and classification, more tahn 30.\
                                \nIt will take a very long time. Probably encoding and restart improve the situation.'


        advice_text = 'ADVICE:  \nYou can try include this feature to argument discrete_columns=[] \
                                \nfor forced classification. Probably it improve precision. \
                                \nOr set discrete_columns="auto"'
        
        
        


        def get_predictors(columns, target_column):
            columns_now = columns[:]
            if target_column in columns:
                columns_now.remove(target_column)
            return columns_now


        def normalize_data(df_in, columns):
            df = df_in.copy()
            for col in columns:
                min_x = df[col].min()
                max_x = df[col].max()
                df[col] = (df[col] - min_x) / (max_x - min_x)
                df[col] = np.log1p(df[col])
            return df

        
        def denormalize_targ(arr, min_y, max_y):
            arr = np.expm1(arr)
            arr = (arr * (max_y - min_y)) + min_y
            return arr
            



        init_time = datetime.datetime.now()

        df = df_0.copy()
        data_info = self.nan_info(df)
        if verbose:
            print('\n\n\n', data_info, '\n\n\n')

        all_features = list(df.columns)
        df_indeces = list(df.index)
        df.reset_index(drop=True, inplace = True)

        all_miss_features = list(data_info.index[data_info['NaN_counts'] > 0])

        # a simple encoding
        df = self.encode_dataframe(df)

        # get continuous & discrete features
        continuous_features = []
        discrete_features = []
        for col in df.columns:
            count_val = len(df[col].value_counts())
            if count_val > CLASS_VALUE_COUNTS:
                continuous_features.append(col)
            else:
                discrete_features.append(col)



        # work with EACH COLUMN containing NaNs
        for target_now in all_miss_features:
            if verbose:
                init_iter_time = datetime.datetime.now()
                print('='*50,'\n')
            # predictors for iteration
            predictors = all_features[:]
            predictors.remove(target_now)

            continuous_features_now = get_predictors(continuous_features, target_now)

            # indexes of missing data in target_now (data for prediction)
            miss_indeces = list((df[pd.isnull(df[target_now])]).index)
            count_miss_values = len(miss_indeces)

            # data without NaN rows (X data for train & evaluation of model)
            work_indeces = list(set(df_indeces) - set(miss_indeces))

            df_normzd = normalize_data(df, continuous_features_now)
            # X data for predict target NaNs
            miss_df = df_normzd.loc[miss_indeces, predictors]
            # X data for train and model evaluation 
            work_df = df_normzd.iloc[work_indeces, : ]
            
            
            target_values_counted = work_df[target_now].value_counts()
            len_target_values_counted = len(target_values_counted)
            feature_type_target = data_info.loc[target_now, 'col_type']
                        
            
            # Info
            if verbose:
                percent_missing_data = data_info.loc[target_now, 'NaN_percent']
                print(f'Feature: {target_now}, missing values: {percent_missing_data}%\n')
            

            # FORK to classification or regression
            classification = False
            to_int_flag = False
            
            if feature_type_target == 'object':
                if len_target_values_counted > 100:
                    classification = False
                    print(colored(target_now, 'red'))
                    print(colored(warning_switch, 'red'))
                    to_int_flag = True
                    
            if isinstance(discrete_columns, list):
                if target_now in discrete_columns:
                    classification = True
                    if len_target_values_counted > CLASS_VALUE_COUNTS:
                        if verbose:
                            print(colored(warning_time, 'yellow'))
                        
                elif len_target_values_counted <= CLASS_VALUE_COUNTS:
                    if verbose:
                        print(colored(advice_text, 'green'))
                        
            if discrete_columns == 'auto':
                if len_target_values_counted <= CLASS_VALUE_COUNTS:
                    if len_target_values_counted <= 10:
                        classification = True
                    elif int(target_values_counted.index.min()) == 0 and target_values_counted.index.max() == len_target_values_counted -1 and len_target_values_counted < 100:
                        classification = True
                    elif int(target_values_counted.index.min()) == 1 and target_values_counted.index.max() == len_target_values_counted  and len_target_values_counted < 100:
                        classification = True
                    

            if classification:
                # Test
                if verbose:
                    print('CLASSIFIER cross validation:')                    
                    test_y_all = np.array([])
                    pred_all = np.array([])
                    
                    k_fold_dict = self.k_folds_split(work_df, target_now, K_FOLDS)
                    for k in range(0, K_FOLDS):
                        lgb_class = lgb.LGBMClassifier(random_state=0, n_jobs=-1, device=device)
                        lgb_class.fit(k_fold_dict['train_X'][k], k_fold_dict['train_y'][k])
                        pred = lgb_class.predict(k_fold_dict['test_X'][k])
                        test_y_all = np.concatenate(([test_y_all, k_fold_dict['test_y'][k]]), axis=None)
                        pred_all = np.concatenate((pred_all, pred), axis=None)


                    if target_now in self.encoder_pool:
                        enc_names = list(self.encoder_pool[target_now].keys())
                        print('\n', classification_report(test_y_all, pred_all, target_names=enc_names), '\n')
                    else:
                        print('\n', classification_report(test_y_all, pred_all), '\n')

                    rng = np.random.default_rng()
                    idx = rng.integers(len(pred_all)-1, size=20)
                    test = np.take(test_y_all, idx)
                    pred = np.take(pred_all, idx)

                    print(f'random 15 y_test: {test[:15]}')
                    print(f'random 15 y_pred: {pred[:15]}\n')



                X = work_df[predictors]
                y = work_df[target_now]    
                
                # Final prediction & imputing
                lgb_class = lgb.LGBMClassifier(random_state=0, n_jobs=-1, device=device)
                lgb_class.fit(X, y)
                pred_miss = lgb_class.predict(miss_df)
                
                if verbose:
                    print(f'first 10 imputed: {np.round(pred_miss[:10], 1)}\n')

                df.loc[miss_indeces, target_now] = np.array(pred_miss)
                counter_predicted_values += len(miss_indeces)


            
            # regression for target_now 
            else:
                min_y = work_df[target_now].min()
                max_y = work_df[target_now].max()
                work_df[target_now] = (work_df[target_now] - min_y) / (max_y - min_y)
                work_df[target_now] = np.log1p(work_df[target_now])
                # Test
                if verbose:
                    print('REGRESSOR cross validation:')

                    test_y_all = np.array([])
                    pred_all = np.array([])
                    
                    k_fold_dict = self.k_folds_split(work_df, target_now, K_FOLDS)
                    for k in range(0, K_FOLDS):
                        lgb_reg = lgb.LGBMRegressor(n_jobs=-1, random_state=0, device=device)
                        lgb_reg.fit(k_fold_dict['train_X'][k], k_fold_dict['train_y'][k])
                        pred = lgb_reg.predict(k_fold_dict['test_X'][k])
                        if min_y >= 0.0:
                            pred[pred < 0] = 0
                        test_y_all = np.concatenate(([test_y_all, k_fold_dict['test_y'][k]]), axis=None)
                        pred_all = np.concatenate((pred_all, pred), axis=None)
                    
                    if to_int_flag or len_target_values_counted <= CLASS_VALUE_COUNTS:
                        pred_all = pred_all.astype(int).astype(float)
                        pred_all[pred_all < min_y] = min_y
                        pred_all[pred_all > max_y] = max_y
                        
                    test_y_all = denormalize_targ(test_y_all, min_y, max_y)
                    pred_all = denormalize_targ(pred_all, min_y, max_y)

                    print(f'                       MAE: {mean_absolute_error(test_y_all, pred_all)}')
                    print(f'                      RMSE: {mean_squared_error(test_y_all,  pred_all) ** 0.5}')

                    print(f'min for {target_now}: {min_y}')
                    print(f'avg for {target_now}: {test_y_all.mean()}')
                    print(f'max for {target_now}: {max_y}\n')

                    rng = np.random.default_rng()
                    idx = rng.integers(len(pred_all)-1, size=10)
                    test = np.take(test_y_all, idx)
                    pred = np.take(pred_all, idx)

                    print(f'random 10 y_test: {list(np.round(test, 1))}')
                    print(f'random 10 y_pred: {list(np.round(pred, 1))}\n')

                X = work_df[predictors]
                y = work_df[target_now]    
                
                # Final prediction & imputing
                lgb_reg = lgb.LGBMRegressor(random_state=0, n_jobs=-1, device=device)
                lgb_reg.fit(X, y)
                pred_miss = lgb_reg.predict(miss_df)
                pred_miss = denormalize_targ(pred_miss, min_y, max_y)
                if min_y >= 0:
                    pred_miss[pred_miss < 0] = 0
                    
                if to_int_flag or len_target_values_counted <= CLASS_VALUE_COUNTS:
                    pred_miss = pred_miss.astype(int).astype(float)
                    pred_miss[pred_miss < min_y] = min_y
                    pred_miss[pred_miss > max_y] = max_y
                    
                if verbose:
                    print(f'first 10 imputed: {list(np.round(pred_miss[:10], 1))}\n')


                df.loc[miss_indeces, target_now] = np.array(pred_miss)
                counter_predicted_values += len(miss_indeces)



            if verbose:
                finish_iter_time = datetime.datetime.now()
                requared = finish_iter_time - init_iter_time
                print(f'Imputed Values: {count_miss_values}')
                print(f'Required time:  {str(requared)}\n')

        # return dataframe states to their initial states (decode, index, types)
        df = self.decode_dataframe(df)

        for col in df.columns:
            df[col] = df[col].astype(data_info.loc[col, 'col_type'])

        df.index = df_indeces

        data_info = self.nan_info(df)
        if verbose:
            print('\n\n\n')
            print(data_info)
            print('\n\n\n')
            print(f'{counter_predicted_values} values have been predicted and replaced. \
            {(counter_predicted_values*100/(df.shape[0]*df.shape[1]))} % of data')
            print('\n')
            finish_time = datetime.datetime.now()
            requared = finish_time - init_time
            print(f'Required time totally: {str(requared)}\n\n')
            
        unprocessed_list = list(data_info[data_info['NaN_counts'].notnull()].index)
        if unprocessed_list:
            text = 'WARNING:\nUnprocessed columns :' + str(unprocessed_list) + '\nYou can try encoding or other methods'
            print(colored(text, 'red'))

        return df