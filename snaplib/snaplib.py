import datetime
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import lightgbm as lgb
import catboost as ctb
import xgboost as xgb


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score











class Snaplib:
    '''
    data preprocessing library
    '''




    def nan_info(self, df):
        '''
        Get pandas.DataFrame with information about missing data in columns

        Use case:
        missing_infodf = Snaplib.nan_info(df)
        '''

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
        Snaplib.nan_plot(df)
        '''

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
        df = Snaplib.cleane(df, targetNone, verbose=True)
        
        '''
        
        # DROP DUPLICATES
        start_shape = df.shape
        if verbose:
            print(f'Start shape: {start_shape}\n\n')
            print(f'DROP DUPLICATES:')
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        dr_dupl_shape = df.shape
        if verbose:
            print(f'{start_shape[0] - dr_dupl_shape[0]} - rows have been dropped')
            print(f'shape: {dr_dupl_shape}\n')

        # DROP COLUMNS with 1 unique value
            print('DROP COLUMNS with 1 unique value:')
        for col in df.columns:
            unique_array = df[col].unique()
            if len(unique_array) == 1:
                df.drop([col], inplace=True, axis=1)
                if verbose:
                    print(f'column "{col}" cnontains 1 unique value -> have been dropped')
            elif len(unique_array) == 2 and np.any(np.isnan(unique_array)):
                if verbose:
                    print(f'!!! column "{col}" cnontains 1 unique value and np.nan')
        if verbose:
            print(f'shape: {df.shape}\n')

        # DROP ROWS with NaN IN TARGET
        if target:
            if verbose:
                print('DROP ROWS with NaN IN TARGET:')
            nan = df[df[target].isnull()]
            indeces = list(nan.index)
            if verbose:
                print(f'{len(indeces)} - rows have been dropped')
            df = df.drop(df.index[indeces])
            df.reset_index(drop=True, inplace=True)
        if verbose:
            print(f'shape: {df.shape}\n')
            print(f'\nFinish shape: {df.shape}\n')
            self.nan_plot(df)
        return df










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
        train_X, test_X, train_y, test_y = Snaplib.train_test_split_balanced(df, target_feature, test_size=0.33, random_state=0, research_iter=0)
        
        1) The input should be a whole DataFrame. It's the first positional argument.
        2) And the second positional argument should be the name of the target feature as a string.
        3) This function has internal testing.
        In this way you can testing the usefulness of the custom split.
        The first test performing with a random_state values from 0 to research_iter argument by sklearn.model_selection.train_test_split.
        Before output performing a final testing.

        You can perform testing by specifying the value of the research_iter argument > 0.

        4) If you are convinced that the function is useful. You can silence the function.
        Set the research_iter arguments to 0 (zero).

        5) The number of possible random_state is an equivalent to 1/test_size.
        6) The necessary libraries are integrated at the beginning of the function.
        '''
        
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
                print(f'Target function has {len(y.value_counts())} unique values.')
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
            train_y = y.iloc[train_indexes]
            test_X = X.iloc[test_indexes]
            test_y = y.iloc[test_indexes]
                    
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
                     stacking = 0, 
                    ):
        ''' 
        Imputing of missing values (np.nan) in tabular data, not TimeSeries.

        Use case:
        df = Snaplib.recover_data(df, verbose=True, stacking=True)

        if set verbose = True algorithm run tests and print results of tests for decision making.
        if set stacking = True algorithm apply ensemble lightgbm, catboost, xgboost, else lightgbm only. 
        And ensemble decrise train/test leakage.
        '''
            
        encoder_pool = {}
        decoder_pool = {}
        encoded_columns = []
        counter_predicted_values = 0
        CLASS_VALUE_COUNTS = 20
        miss_indeces = None


        
        
        def encode_column(df, enc_pool, dec_pool, column):
            enc_pool[column] = {}
            dec_pool[column] = {}
            not_nan_index=df[df[column].notnull()].index
            values_set = list(set(list(df.loc[not_nan_index, column])))
            value = 0.0
            for el in values_set:
                enc_pool[column][el] = value
                dec_pool[column][value] = el
                value += 1
            df[column] = df[column].map(enc_pool[column])
            df[column] = df[column].astype('float64')
            return df
        
        
        def get_columns(columns, target_column):
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
        
        
        def get_class_weight(y):
            global class_weight
            class_weight={}
            unique = []
            counts = []
            values_counted = (pd.DataFrame(y)).value_counts().sort_values(ascending=False)
            for idx, val in values_counted.items():
                unique.append(idx[0])
                counts.append(val)

            max_value = max(counts)
            j = 0
            for el in unique:
                class_weight[el] = int(max_value/counts[j])
                j += 1
            if len(unique) == 1:
                class_weight = {}
            return class_weight

        
        def predict(X__train, y__train, X__pred, all_algorithms, values_counted=None):
            if len(pd.Series(y__train).value_counts()) == 1:
                all_algorithms = [all_algorithms[0]]
                
            stacked_predicts = pd.DataFrame()
            stacked_column_names = []
            for alg in all_algorithms:
                alg_name = str(alg.__class__.__name__)[:3]
                model = alg.fit(X__train, y__train.ravel())
                y_hat = model.predict(X__pred).ravel()
                stacked_predicts[alg_name] = y_hat
                stacked_column_names.append(alg_name)
            if values_counted:
                # classification
                stacked_predicts['y_hat_final'] = stacked_predicts[stacked_column_names].mode(axis=1)[0].astype('int64')
                y_hat = list(stacked_predicts.loc[:, 'y_hat_final'])
            else:
                # regression
                stacked_predicts['y_hat_final'] = stacked_predicts[stacked_column_names].mean(axis=1)
                y_hat = stacked_predicts.loc[:, 'y_hat_final']
                y_hat[y_hat == -np.inf] = 0
                y_hat[y_hat == np.inf] = 0
    #         print(stacked_predicts)
            del stacked_predicts
            return y_hat

        
        def imput_missing_value_to_main_df(df, miss_indeces, pred_miss, el):
            df.loc[miss_indeces, el] = pred_miss[:]
            return df
        
        
        def decode_column(df, dec_pool, column):
            df[column] = df[column].map(dec_pool[column])
            df[column] = df[column].astype('object')
            return df
        
        
        
        
        # main
        init_time = datetime.datetime.now()
        
        df = df_0.copy()

        if verbose:
            self.nan_plot(df)
                
        data_info = self.nan_info(df)
        if verbose:
            print('\n\n\n', data_info, '\n\n\n')
        
        all_features = list(df.columns)
        df_indeces = list(df.index)
        df.reset_index(drop=True, inplace = True)
        
        all_miss_features = list(data_info.index[data_info['NaN_counts'] > 0])
        
                
        # a simple encoding
        for col in df.columns:
            feature_type = data_info.loc[col, 'col_type']
            if feature_type == 'object' or feature_type == 'bool':            
                df[col] = encode_column(df[[col]], encoder_pool, decoder_pool, col)
                encoded_columns.append(col)
            else:
                # object column with NaN sometimes has type float64
                try:
                    df.loc[:, col] = df.loc[:, col] + 0
                except:
                    df[col] = df[col].astype('object')
                    df[col] = encode_column(df[[col]], encoder_pool, decoder_pool, col)
                    encoded_columns.append(col)
        
        
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
                print('='*90,'\n')
            # predictors for iteration
            predictors = all_features[:]
            predictors.remove(target_now)
            
            continuous_features_now = get_columns(continuous_features, target_now)
            # discrete_features_now = get_columns(discrete_features, target_now)
            
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
            else:
                # normalization
                min_y = y.min()
                max_y = y.max()
                y = (y - min_y) / (max_y - min_y)
                y = np.log1p(y)

            
                

            # split for testing
            if feature_type_target == 'object' and last_item < 2:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
            elif feature_type_target == 'object' and last_item < 3:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
            elif feature_type_target == 'object' and last_item < 5:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
            elif feature_type_target == 'object':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # df_conctd = pd.concat([X, y], axis=1)
            # X_train, X_test, y_train, y_test = self.train_test_split_balanced(df_conctd, target_now, test_size=0.2, random_state=0, research_iter=0)

            
            # Info
            if verbose:
                percent_missing_data = data_info.loc[target_now, 'NaN_percent']
                print(f'Feature: {target_now}, missing values: {percent_missing_data}%\n')    
                print(f'Shape for train & test: {(X.shape)}')
                print('')
                if len_target_values_counted < 11 or feature_type_target == 'object':
                    print(f'values_count:')
                    print(target_values_counted)
                    print('')

            
            # PREDICTIONS CLASSIFIER
            if len_target_values_counted < CLASS_VALUE_COUNTS or feature_type_target == 'object':
                class_weight = get_class_weight(y_train)
                lgb_class = lgb.LGBMClassifier(class_weight=class_weight, random_state=0, n_jobs=-1)
                algorithms = [lgb_class]
                if stacking:
                    xgb_class = xgb.XGBClassifier(sample_weight=class_weight, random_state=0, n_jobs=-1, verbosity=0)
                    ctb_class = ctb.CatBoostClassifier(class_weights=class_weight, random_state=0, verbose=0)
                    algorithms = [lgb_class, xgb_class, ctb_class]
                    
                pred_test = predict(X_train, y_train, X_test, algorithms, len_target_values_counted)

                if verbose:
                    print('CLASSIFIER:')
                    print(f'Weights transferred to the classifier: {class_weight}')
                    print('\nEvaluations:')
                    print(f'first 20 y_test: {list(y_test[:20])}')
                    print(f'first 20 y_pred: {pred_test[:20]}\n')
                    print(f'Classification Report:\n')
                    print(classification_report(y_test, pred_test), '\n')

    #             ADVANCED  alg = LGBMClassifier(n_jobs=-1, random_state=0, class_weight=class_weight)                
                pred_miss = predict(X_train, y_train, miss_df, algorithms, len_target_values_counted)

                pred_miss = [int(i) for i in pred_miss]
                pred_miss = labelencoder.inverse_transform(pred_miss)
                imput_missing_value_to_main_df(df, miss_indeces, pred_miss, target_now)
                counter_predicted_values += len(miss_indeces)
            
            # PREDICTIONS REGRESSOR
            elif feature_type_target == 'float64' or feature_type == 'int64':
                lgb_reg = lgb.LGBMRegressor(n_jobs=-1, random_state=0)
                algorithms = [lgb_reg]
                if stacking:
                    xgb_reg = xgb.XGBRegressor(n_jobs=-1, random_state=0, verbosity=0)
                    ctb_reg = ctb.CatBoostRegressor(random_state=0, verbose=0)
                    algorithms = [lgb_reg, xgb_reg, ctb_reg]
        
                pred_test = predict(X_train, y_train, X_test, algorithms)
                pred_test = np.expm1(pred_test)
                pred_test = (pred_test * (max_y - min_y)) + min_y
                y_test = (y_test * (max_y - min_y)) + min_y


                MAE = mean_absolute_error(y_test,pred_test)
                y_test = list(np.round(y_test[:10], 1))
                y_pred = list(np.round(pred_test[:10], 1))    ##############
                
                if verbose:
                    print('REGRESSOR:')
                    print('\nEvaluations:')
                    print(f'first 10 y_test: {y_test}')
                    print(f'first 10 y_pred: {y_pred}\n')
                    print(f'MAE for {target_now}: {MAE}')
                    print(f'min for {target_now}: {df[target_now].min()}')
                    print(f'avg for {target_now}: {df[target_now].mean()}')
                    print(f'max for {target_now}: {df[target_now].max()}\n')
                    


    #             ADVANCED  alg = LGBMRegressor(n_jobs=-1, random_state=0)        
                pred_miss = predict(X, y, miss_df, algorithms)
                pred_miss = np.expm1(pred_miss)
                pred_miss = (pred_miss * (max_y - min_y)) + min_y
                
                
                imput_missing_value_to_main_df(df, miss_indeces, list(pred_miss), target_now)
                counter_predicted_values += len(miss_indeces)
                
                

            else:
                if verbose:
                    print(f"unprocessed feature: {target_now} - {feature_type} type")


            
            del predictors
            del miss_indeces
            del miss_df        
            del work_df
            del X
            del y
            del X_train
            del X_test
            del y_train
            del y_test
            
            if verbose:
                finish_iter_time = datetime.datetime.now()
                requared = finish_iter_time - init_iter_time
                print(f'Imputed Values: {count_miss_values}')
                print(f'Required time:  {str(requared)}\n')

            
        # return states to their initial states
        for col in encoded_columns:
            df[col] = decode_column(df[[col]], decoder_pool, col)
            
        for col in df.columns:
            df[col] = df[col].astype(data_info.loc[col, 'col_type'])
                    
        df.index = df_indeces

        if verbose:
            self.nan_plot(df)
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