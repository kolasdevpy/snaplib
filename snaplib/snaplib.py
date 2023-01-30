import datetime
import sys
import pickle
from os import makedirs
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns




import lightgbm as lgb
import catboost as ctb
import xgboost as xgb


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score











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
        values_set = list(set(list(df.loc[not_nan_index, column])))
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
    
    
    
    
    
    
    



    def encode_dataframe(self, df):
        '''
        encode a dataframe

        Use case:
        df = Snaplib().encode_dataframe(df)
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')

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
                                                                              research_iter=0
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
                            X_pred, 
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
                                        X_pred, 
                                        y_test or None, 
                                        task='clsf' or 'regr', 
                                        verbose= 0, 1, 2
                                        ):
        
        algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        X_train and y_train are data for training list of algorithms.
        X_pred is dataframe for prediction.
        y_test optionaly. If exist visualize it as last column on a plot (verbose=True). 
        task='clsf' or 'regr', classification or regression
        verbose = 0 mute, 1 verbose.

        '''
        if type(algorithms_list) != list:
            raise TypeError('algorithms_list must be of list type.')
        if not isinstance(X_train, pd.core.frame.DataFrame):
            raise TypeError('The X__train must be a pandas.core.frame.DataFrame instance.')
        if not isinstance(X_pred, pd.core.frame.DataFrame):
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
        for alg in algorithms_list:
            alg_name = alg.__class__.__name__[:3]
            # if y_test is not None and alg_name in ['LGB', 'XGB', 'Cat']:
            #     model = alg.fit(X_train, y_train, eval_set=[(X_pred, y_test)], early_stopping_rounds=10, verbose=False)
            # else:
            model = alg.fit(X_train, y_train)

            y_hat = model.predict(X_pred)
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



        y_hat = stacked_predicts.loc[:, 'Y_HAT_STACKED']
        if verbose:
            plt.figure(figsize=(5, 10))
            sns.heatmap(stacked_predicts[-1000:], cbar=False)
            plt.show()
        return y_hat


    
    
    
    
    
    
    



    def cross_val(self, algorithms, k_fold_dict, metric, task, cv, verbose=0):
        
        ''' 
        Cross Validation method for list of algorithms.
        
        Use case:
        y_hat = Snaplib().cross_val(algorithms, k_fold_dict, metric, task, cv, verbose=0):
        
        algorithms_list = list of algorithms like [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
        k_fold_dict is a dictionary with the structure:
        
        K_FOLD = 3
        k_fold_dict = { 
                        'train_X' : [df_0, df_1, df_2],
                        'test_X'  : [df_0, df_1, df_2],
                        'train_y' : [seri_0, seri_1, seri_2],
                        'test_y'  : [seri_0, seri_1, seri_2],
                       }
              
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


        results=[]
        test_y_all = np.array([])
        pred_all = np.array([])
        if task == 'clsf':
            length = len(k_fold_dict['train_y'][0].value_counts().index)
            cm_base = np.zeros([length, length])
            # cm_base = np.array([[0, 0], [0, 0]])


        for k in range(0, cv):
            if len(algorithms) > 1:
                pred = self.fit_predict_stacked(algorithms, 
                                                k_fold_dict['train_X'][k], 
                                                k_fold_dict['train_y'][k], 
                                                k_fold_dict['test_X'][k], 
                                                k_fold_dict['test_y'][k], 
                                                task,
                                                verbose
                                            )
            else:
                alg = algorithms[0]
                alg.fit(k_fold_dict['train_X'][k], k_fold_dict['train_y'][k])
                pred = alg.predict(k_fold_dict['test_X'][k])

            if verbose:
                if task == 'clsf':
                    cm = confusion_matrix(k_fold_dict['test_y'][k], pred)
                    cm_base = cm_base + cm

            test_y_all = np.concatenate(([test_y_all, k_fold_dict['test_y'][k]]), axis=None)
            pred_all = np.concatenate((pred_all, pred), axis=None)

            score = metric(k_fold_dict['test_y'][k], pred)
            results.append(score)

        results_std = np.std(results)
        score = sum(results) / len(results)

        if verbose:
            for alg in algorithms:
                print(alg.__class__.__name__)
                
            print('')
            if str(metric).split('.')[0] == 'functools':
                metric_name = str(metric).split('function ')[1].split(' ')[0]
            else:
                metric_name = metric.__name__
            print("%s %0.6f (std: +/- %0.2f)" % (metric_name, score, results_std))
            print('\n', results, '\n')

            if task == 'clsf':
                print('\n', classification_report(test_y_all, pred_all), '\n')
                plt.figure(figsize=(3, 3))
                sns.heatmap(cm_base, annot=True, cmap="Blues", fmt='.0f',  cbar=False)
                plt.show()

        return score
    
    








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










    def features_selection_clsf(self, algorithms, df, target, metric, cv, verbose=0):
        

        ''' 
        Select bests features for modeling and return list with bad features required to be droped.
        
        Use case:
        features_to_drop = Snaplib().features_selection_clsf(self, algorithms, df, target, metric, cv, verbose=0):
        df.drop(features_to_drop, inplace=True, axis=1)


        algorithms_list = list of algorithms like [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
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
        score_with_all_features  = self.cross_val(algorithms, k_folds_dict_data, metric, task='clsf', cv=cv, verbose=0)
        if verbose:
            print(f'{score_with_all_features}     General cv_score with all features')
        
        
        while score_with_all_features <= cv_score_of_bad_feature:
            scores_df = pd.DataFrame(index = features, columns=['cv_score'])
            k_folds_dict_data = self.k_folds_split(df[features], target, cv)
            score_with_all_features  = self.cross_val(algorithms, k_folds_dict_data, metric, task='clsf', cv=cv, verbose=0)
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
                    score  = self.cross_val(algorithms, k_folds_dict_data, metric, task='clsf', cv=cv, verbose=0)
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
                    print('--------------------------------------------')
                    print(f'    APPEND TO DROP    {bad_feature}')
                    print('--------------------------------------------')
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
                                  research_iter=0
                                  ):
        ''' 
        Split the data with the distribution as close as possible 
        to the same in both the train set and the test set, and not only for the target column, 
        but also for all other columns.

        Use case:
        train_X, test_X, train_y, test_y = Snaplib().train_test_split_balanced(df, target_name, test_size=0.2, random_state=0, research_iter=0)
        
        1) The input should be a whole DataFrame. It's the first positional argument.
        2) And the second positional argument should be the name of the target feature as a string.
        3) This method has internal testing.
        In this way you can testing the usefulness of the custom split.
        The first test performing with a random_state values from 0 to research_iter argument by sklearn.model_selection.train_test_split.
        Before output performing a final testing.

        You can perform testing by specifying the value of the research_iter argument > 0.

        4) If you are convinced that the method is useful. You can silence the method.
        Set the research_iter arguments to 0 (zero).

        5) The number of possible random_state is an equivalent to 1/test_size.
        6) The necessary libraries are integrated at the beginning of the method.
        '''

        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(target_feature) != str:
            raise TypeError('target_feature must be of str type.')
        if type(test_size) != float:
            raise TypeError('test_size must be of float type in [0.0, 1.0].')
        if type(random_state) != int:
            raise TypeError('random_state must be of int type.')
        if type(research_iter) != int:
            raise TypeError('research_iter must be of int type. Recomended interval [0:100]')

        
        CLASSIFIER_FOR_UNIQUE_VALUES_LESS_THAN = 20
        df_count = pd.DataFrame()
        
        
        def get_predictors(df, target_feature):
            predictors = list(df.columns)
            predictors.remove(target_feature)
            return predictors
        
        
        def get_X(df, predictors):
            X = df[predictors]
            return X
        
        
        def get_y(df, target_feature):
            y = df[[target_feature]]
            return y
        
        
        def regression_score(train_X, test_X, train_y, test_y):
            model = lgb.LGBMRegressor(random_state=0).fit(train_X, train_y)
            predict = model.predict(test_X)
            return mean_absolute_error(predict, test_y)
        
        
        def classification_accuracy(train_X, test_X, train_y, test_y):
            model = lgb.LGBMClassifier(random_state=0).fit(train_X, train_y.values.ravel())
            predict = model.predict(test_X)
            return f1_score(predict, test_y.values.ravel(), average='macro')
        
        
        def get_research(X, y, target_feature, test_size, research_iter):
            RESULTS = pd.DataFrame()
            if len(y.value_counts()) > CLASSIFIER_FOR_UNIQUE_VALUES_LESS_THAN:
                print('Regression test by sklearn.model_selection.train_test_split:\n')
                for random_state in range(0,research_iter):
                    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
                    RESULTS.loc[random_state, 'score'] = regression_score(train_X, test_X, train_y, test_y)
                print(f'Regression MAE with random_state from 0 to {research_iter - 1}:')
            else:
                print('Classification test by sklearn.model_selection.train_test_split  with stratify=y:\n')
                print(f'Target feature has {len(y.value_counts())} unique values.')
                for random_state in range(0,research_iter):
                    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
                    RESULTS.loc[random_state, 'score'] = classification_accuracy(train_X, test_X, train_y, test_y)
                print(f'classification F1_SCORE(average="macro") with random_state from 0 to {research_iter - 1}:')

            print(f'max:  {RESULTS.score.max()}')
            print(f'mean: {RESULTS.score.mean()}')
            print(f'min:  {RESULTS.score.min()}\n')
            del RESULTS
            return 0
            
        
        def order_and_sort_table(df, important_functions):
            df = df.sort_values(by=important_functions, ascending=True)
            df = df.reset_index(drop=True)
            if research_iter:
                print('\n-----------------------------------')
                print(f'\nThe Table has been ordered and sorted by columns:\n\n{important_functions}')
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
                    
    #         train_indexes = list(filter(lambda x: (x + increment) % every_n_el, X.index))
    #         test_indexes = list(filter(lambda x: not (x + increment) % every_n_el, X.index))

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
        init_time = datetime.datetime.now()
        predictors = get_predictors(df, target_feature)
        if research_iter:
            get_research(get_X(df, predictors), get_y(df, target_feature), target_feature, test_size, research_iter)
        
        for el in df.columns:
            count = len(df[el].value_counts())
            df_count.loc[el, 'counts'] = count
        
        columns = list(df_count.columns)
        df_count = df_count.sort_values(by=columns, ascending=True)

        if research_iter:
            print('\n-----------------------------------')
            print(df_count)
        
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
        train_test_split_ordered(get_X(df, predictors), get_y(df, target_feature), random_state, test_size=test_size)

        if research_iter:
            print('\n-----------------------------------')
            if  len(train_y.value_counts()) > CLASSIFIER_FOR_UNIQUE_VALUES_LESS_THAN:
                print(f'The final result MAE of the custom split:\n\
                \nMAE:  {regression_score(train_X, test_X, train_y, test_y)}')
            else:
                print(f'The final result F1_SCORE(average="macro") of the custom split:\n\nF1_SCORE: \
                {classification_accuracy(train_X, test_X, train_y, test_y)}')
        if research_iter:
            print('\n-----------------------------------')
            finish_time = datetime.datetime.now()
            requared_time = finish_time - init_time
            print(f'Required time:  {str(requared_time)}\n')
            
        if research_iter:
            print('\n===============   DISTRIBUTIONS   ===============\n\n')
            visualize(train_y, test_y, target_feature, ' train_y_TARGET', ' test_y_TARGET')
            for column in train_X.columns:
                visualize(train_X[column], test_X[column], column, ' train_X', ' test_X')

        return train_X, test_X, train_y, test_y










    def recover_data(self,
                     df_0, 
                     verbose = 1,
                     ):
        ''' 
        Imputing of missing values (np.nan) in tabular data, not TimeSeries.

        Use case:
        df = Snaplib().recover_data(df, verbose=True, stacking=True)

        if set verbose = if True algorithm runs cross validation tests and print results of tests for decision making.
        And ensemble decrise train/test leakage.
        '''

        if not isinstance(df_0, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
        if type(verbose) != int and type(verbose) != bool:
            raise TypeError('verbose must be of int type or bool.')


        counter_predicted_values = 0
        CLASS_VALUE_COUNTS = 30
        K_FOLDS = 3
        miss_indeces = None


        def get_predictors(columns, target_column):
            columns_now = columns[:]
            if target_column in columns:
                columns_now.remove(target_column)
            return columns_now


        def normalize_data(df_in, columns):
            for col in columns:
                min_x = df_in[col].min()
                max_x = df_in[col].max()
                df_in[col] = (df_in[col] - min_x) / (max_x - min_x)
                df_in[col] = np.log1p(df_in[col])
            return df_in




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


        # work with each column containing NaNs
        for target_now in all_miss_features:
            if verbose:
                init_iter_time = datetime.datetime.now()
                print('='*50,'\n')
            # predictors for iteration
            predictors = all_features[:]
            predictors.remove(target_now)

            continuous_features_now = get_predictors(continuous_features, target_now)
            # discrete_features_now = get_predictors(discrete_features, target_now)

            # indexes of missing data in target_now (data for prediction)
            miss_indeces = list((df[pd.isnull(df[target_now])]).index)
            count_miss_values = len(miss_indeces)

            # data without NaN rows (X data for train & evaluation of model)
            work_indeces = list(set(df_indeces) - set(miss_indeces))

            # X data for predict target NaNs
            miss_df = df.loc[miss_indeces, predictors]
            miss_df = normalize_data(miss_df, continuous_features_now)

            # X data for train and model evaluation 
            work_df = df.iloc[work_indeces, : ]
            work_df = normalize_data(work_df, continuous_features_now)

            X = work_df[predictors]
            y = work_df[target_now]
            y[y == -np.inf] = 0
            y[y == np.inf] = 0


            target_values_counted = y.value_counts()
            last_item = target_values_counted.tail(1).item()
            len_target_values_counted = len(target_values_counted)


            feature_type_target = data_info.loc[target_now, 'col_type']


            if len_target_values_counted <= CLASS_VALUE_COUNTS or feature_type_target == 'object':
                labelencoder = LabelEncoder()
                y = labelencoder.fit_transform(y).astype('int64')
                work_df[target_now] = work_df[target_now].astype('int64')
            else:
                # normalization
                min_y = y.min()
                max_y = y.max()
                y = (y - min_y) / (max_y - min_y)
                y = np.log1p(y)

            # Info
            if verbose:
                percent_missing_data = data_info.loc[target_now, 'NaN_percent']
                print(f'Feature: {target_now}, missing values: {percent_missing_data}%\n')
                # split for testing
                k_fold_dict = self.k_folds_split(work_df, target_now, K_FOLDS)

            # PREDICTIONS CLASSIFIER
            if len_target_values_counted < CLASS_VALUE_COUNTS or feature_type_target == 'object':
                # Test
                if verbose:
                    print('CLASSIFIER cross validation:')                    
                    results=[]
                    test_y_all = np.array([])
                    pred_all = np.array([])

                    for k in range(0, K_FOLDS):
                        lgb_class = lgb.LGBMClassifier(random_state=0, n_jobs=-1)
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
                    
                    print(f'first 20 y_test: {test[:20]}')
                    print(f'first 20 y_pred: {pred[:20]}\n')
                    
                    

                # Final prediction
                lgb_class = lgb.LGBMClassifier(random_state=0, n_jobs=-1)
                lgb_class.fit(X, y)
                pred_miss = lgb_class.predict(miss_df)

                pred_miss = labelencoder.inverse_transform(pred_miss)

                df.loc[miss_indeces, target_now] = np.array(pred_miss)
                counter_predicted_values += len(miss_indeces)

            # PREDICTIONS REGRESSOR
            elif feature_type_target == 'float64' or feature_type_target == 'int64':
                # Test
                if verbose:
                    print('REGRESSOR cross validation:')

                    results=[]
                    test_y_all = np.array([])
                    pred_all = np.array([])
                    for k in range(0, K_FOLDS):
                        lgb_reg = lgb.LGBMRegressor(n_jobs=-1, random_state=0)
                        lgb_reg.fit(k_fold_dict['train_X'][k], k_fold_dict['train_y'][k])
                        pred = lgb_reg.predict(k_fold_dict['test_X'][k])
                        if y.min() == 0.0:
                            pred[pred < 0] = 0
                        test_y_all = np.concatenate(([test_y_all, k_fold_dict['test_y'][k]]), axis=None)
                        pred_all = np.concatenate((pred_all, pred), axis=None)

                    print(f'                   MAE: {mean_absolute_error(test_y_all, pred_all)}')
                    print(f'                  RMSE: {np.sqrt(((test_y_all - pred_all) ** 2).mean())}')

                    print(f'min for {target_now}: {df[target_now].min()}')
                    print(f'avg for {target_now}: {df[target_now].mean()}')
                    print(f'max for {target_now}: {df[target_now].max()}\n')
                    
                    rng = np.random.default_rng()
                    idx = rng.integers(len(pred_all)-1, size=10)
                    test = np.take(test_y_all, idx)
                    pred = np.take(pred_all, idx)
                    
                    print(f'first 10 y_test: {list(np.round(test, 1))}')
                    print(f'first 10 y_pred: {list(np.round(pred, 1))}\n')

                # Final prediction
                lgb_reg = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
                lgb_reg.fit(X, y)
                pred_miss = lgb_reg.predict(miss_df)
                if y.min() == 0:
                    pred_miss[pred_miss < 0] = 0

                pred_miss = np.expm1(pred_miss)
                pred_miss = (pred_miss * (max_y - min_y)) + min_y

                df.loc[miss_indeces, target_now] = np.array(pred_miss)
                counter_predicted_values += len(miss_indeces)

            else:
                if verbose:
                    print(f"unprocessed feature: {target_now} - {feature_type_target} type")

            if verbose:
                finish_iter_time = datetime.datetime.now()
                requared = finish_iter_time - init_iter_time
                print(f'Imputed Values: {count_miss_values}')
                print(f'Required time:  {str(requared)}\n')

        # return dataframe state to their initial states (decode, index, types)
        df = self.decode_dataframe(df)

        for col in df.columns:
            df[col] = df[col].astype(data_info.loc[col, 'col_type'])

        df.index = df_indeces

        if verbose:
            print('\n\n\n')
            data_info = self.nan_info(df)
            print(data_info)
            print('\n\n\n')
            print(f'{counter_predicted_values} values have been predicted and replaced. \
            {(counter_predicted_values*100/(df.shape[0]*df.shape[1]))} % of data')
            print('\n')
            finish_time = datetime.datetime.now()
            requared = finish_time - init_time
            print(f'Required time totally: {str(requared)}\n\n')

        return df