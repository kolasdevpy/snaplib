import datetime
import pandas as pd
import numpy as np
from termcolor import colored

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, \
                            mean_squared_error, \
                            classification_report


from . import counter
from . import encoder
from . import nans
from . import splitter










def recover_data(df_0 : pd.DataFrame, 
                 device : str = 'cpu',
                 verbose : int or bool = True,
                 discrete_columns : list or str = 'auto', 
                 ) -> pd.DataFrame:
    ''' 
    Imputing of missing values (np.nan) in tabular data, not TimeSeries.

    Use case:
    df = Snaplib().recover_data(df, device="cpu", verbose=True)
    device must be "cpu" or "gpu". Sometime small datasets work faster with cpu.
    verbose = True algorithm runs cross validation tests and prints results of tests for decision making.
    discrete_columns = ['col_name_1', 'col_name_2', 'col_name_3', 'etc']

    TESTS on https://www.kaggle.com/code/artyomkolas/nan-prediction-in-progress/notebook
    '''

    if not isinstance(df_0, pd.core.frame.DataFrame):
        raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
    if device !='cpu' and device != 'gpu':
        raise ValueError('device must be "cpu" or "gpu".')
    if type(verbose) != int and type(verbose) != bool:
        raise TypeError('verbose must be of int type or bool.')
    if not isinstance(discrete_columns, list) and discrete_columns != 'auto':
        raise TypeError('The discrete_columns must be a list instance or string "auto".')



    counter_predicted_values = counter.Counter(start=0)
    CLASS_VALUE_COUNTS = 20
    K_FOLDS = 4
    ENCODER = encoder.Encoder()


    warning_switch = 'WARNING:\nYou have a lot of unique values for this discrete column, more tahn 100.\
                                \nPrediction and imputing has been switched to regression. \
                                \nTry encode this column and restart this algorithm.\
                                \nOtherwise you can include this feature to argument discrete_columns=[] \
                                \nfor forced classification. It will take a very long time.\n'

    warning_time = 'WARNING:\nYou have a lot of unique values for this column and classification, more tahn 50.\
                            \nIt will take a very long time. Probably encoding and restart improve the situation.'


    advice_text = 'ADVICE:  \nYou can try include this feature to argument discrete_columns=[] \
                            \nfor forced classification. Probably it improve precision. \
                            \nOr set discrete_columns="auto"'




    def get_predictors(columns, target_column):
        columns_now = columns[:]
        if target_column in columns:
            columns_now.remove(target_column)
        return columns_now
    
    
    def chek_int(ls):
        set_ls = list(set([el.split('.')[1] for el in ls]))
        if len(set_ls) == 1 and set_ls[0] == '0':
            condition = True
        else:
            condition = False
        return condition
    

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
    data_info = nans.nan_info(df)
    if verbose:
        print('\n\n\n', data_info, '\n\n\n')

    all_features = list(df.columns)
    df_indeces = list(df.index)
    df.reset_index(drop=True, inplace = True)

    all_miss_features = list(data_info.index[data_info['NaN_counts'] > 0])

    # a simple encoding
    df = ENCODER.encode_dataframe(df)

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
                if len_target_values_counted > 50:
                    if verbose:
                        print(colored(warning_time, 'yellow'))

            elif len_target_values_counted <= CLASS_VALUE_COUNTS:
                if verbose:
                    print(colored(advice_text, 'green'))

        if discrete_columns == 'auto':
            str_floats = np.array(work_df[target_now].value_counts(dropna=False).index).astype(str).tolist()
            ints = chek_int(str_floats)
            if len_target_values_counted <= CLASS_VALUE_COUNTS and ints:
                classification = True

        if classification:
            # Test
            if verbose:
                print('CLASSIFIER cross validation:')                    
                test_y_all = np.array([])
                pred_all = np.array([])

                k_fold_dict = splitter.k_folds_split(work_df, target_now, K_FOLDS)
                for k in range(0, K_FOLDS):
                    lgb_class = lgb.LGBMClassifier(random_state=0, n_jobs=-1, device=device)
                    lgb_class.fit(k_fold_dict['train_X'][k], k_fold_dict['train_y'][k])
                    pred = lgb_class.predict(k_fold_dict['test_X'][k])
                    test_y_all = np.concatenate(([test_y_all, k_fold_dict['test_y'][k]]), axis=None)
                    pred_all = np.concatenate((pred_all, pred), axis=None)


                if target_now in ENCODER.encoder_pool:
                    enc_names = list(ENCODER.encoder_pool[target_now].keys())
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
                print(f'first 15 imputed: {np.round(pred_miss[:15], 1)}\n')

            df.loc[miss_indeces, target_now] = np.array(pred_miss)
            counter_predicted_values(len(miss_indeces))



        # regression for target_now 
        else:
            # str_floats = np.array(work_df[target_now].value_counts(dropna=False).index).astype(str).tolist()
            # to_int_flag = chek_int(str_floats)
            # print('to_int_flag', to_int_flag)

            min_y = work_df[target_now].min()
            max_y = work_df[target_now].max()
            work_df[target_now] = (work_df[target_now] - min_y) / (max_y - min_y)
            work_df[target_now] = np.log1p(work_df[target_now])
            # Test
            if verbose:
                print('REGRESSOR cross validation:')

                test_y_all = np.array([])
                pred_all = np.array([])

                k_fold_dict = splitter.k_folds_split(work_df, target_now, K_FOLDS)
                for k in range(0, K_FOLDS):
                    lgb_reg = lgb.LGBMRegressor(n_jobs=-1, random_state=0, device=device)
                    lgb_reg.fit(k_fold_dict['train_X'][k], k_fold_dict['train_y'][k])
                    pred = lgb_reg.predict(k_fold_dict['test_X'][k])

                    test_y_all = np.concatenate(([test_y_all, k_fold_dict['test_y'][k]]), axis=None)
                    pred_all = np.concatenate((pred_all, pred), axis=None)

                test_y_all = denormalize_targ(test_y_all, min_y, max_y)
                pred_all = denormalize_targ(pred_all, min_y, max_y)
                
                if to_int_flag:
                    pred_all = np.round(pred_all, 0)
                pred_all[pred_all < min_y] = min_y
                # pred_all[pred_all > max_y] = max_y
                
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

            if to_int_flag:
                pred_miss = np.round(pred_miss, 0)
            pred_miss[pred_miss < min_y] = min_y
            # pred_miss[pred_miss > max_y] = max_y

            if verbose:
                print(f'first 10 imputed: {list(np.round(pred_miss[:10], 1))}\n')

            df.loc[miss_indeces, target_now] = np.array(pred_miss)
            counter_predicted_values(len(miss_indeces))

        if verbose:
            finish_iter_time = datetime.datetime.now()
            requared = finish_iter_time - init_iter_time
            print(f'Imputed Values: {count_miss_values}')
            print(f'Required time:  {str(requared)}\n')

    # return dataframe states to their initial states (decode, index, types)
    df = ENCODER.decode_dataframe(df)

    for col in df.columns:
        df[col] = df[col].astype(data_info.loc[col, 'col_type'])

    df.index = df_indeces

    data_info = nans.nan_info(df)
    if verbose:
        print('\n\n\n')
        print(data_info)
        print('\n\n\n')
        print(f'{counter_predicted_values.counter} values have been predicted and replaced. \
        {(counter_predicted_values.counter*100/(df.shape[0]*df.shape[1]))} % of data')
        print('\n')
        finish_time = datetime.datetime.now()
        requared = finish_time - init_time
        print(f'Required time totally: {str(requared)}\n\n')

    unprocessed_list = list(data_info[data_info['NaN_counts'].notnull()].index)
    if unprocessed_list:
        text = 'WARNING:\nUnprocessed columns: ' + str(unprocessed_list) + '\nYou can try encoding or other methods'
        print(colored(text, 'red'))

    return df