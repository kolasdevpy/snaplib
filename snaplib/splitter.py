import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import lightgbm as lgb






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
        train_X, test_X, train_y, test_y = train_test_split_balanced(df, 
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










def train_test_split_balanced(  df : pd.DataFrame, 
                                target_feature : str, 
                                test_size : float, 
                                random_state: int, 
                                research : bool, 
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