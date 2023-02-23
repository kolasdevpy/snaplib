import pandas as pd
from typing import Callable

from . import splitter
from . import cross_validation




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
    features_to_drop = Snaplib().features_selection_regr(algorithms, df, target, metric, cv, verbose=0):
    df.drop(features_to_drop, inplace=True, axis=1)


    algorithms is a list of algorithms like algs = [
                                                    [LGBMRegressor, dict(params)],
                                                    [XGBRegressor, dict(params)], 
                                                    [CatBoostRegressor, dict(params)],
                                                    ]

    df = pandas.core.frame.DataFrame.
    target = name of target of str type.

    metric is a metric like f1_score or mean_absolute_error.
    cv is num K_FOLD integer 
    verbose = 0 mute, 1 verbose.
    early_stopping_rounds default 0 or positive int.
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

    k_folds_dict_data = splitter.k_folds_split(df[features], target, cv)
    score_with_all_features  = cross_validation.cross_val(
                                    algorithms,
                                    k_folds_dict_data, 
                                    metric, 
                                    task='regr', 
                                    cv=cv, 
                                    verbose=0,
                                    early_stopping_rounds=early_stopping_rounds)
    if verbose:
        print(f'{score_with_all_features}     General cv_score with all features')


    while score_with_all_features >= cv_score_of_bad_feature:
        scores_df = pd.DataFrame(index = features, columns=['cv_score'])
        k_folds_dict_data = splitter.k_folds_split(df[features], target, cv)
        score_with_all_features  = cross_validation.cross_val(
                                        algorithms,
                                        k_folds_dict_data, 
                                        metric, 
                                        task='regr', 
                                        cv=cv, 
                                        verbose=0,
                                        early_stopping_rounds=early_stopping_rounds)
        if verbose:
            print('\n\n')
            print(f'{len(features)} number of features')
            print("{:1.8f}   {:20}  ".format(score_with_all_features, 'BASE cv_score with all features'))
            print('\n\nwithou feature\n')

        for without_feature in features:
            if without_feature != target:
                fit_faetures = features[:]
                fit_faetures.remove(without_feature)
                k_folds_dict_data = splitter.k_folds_split(df[fit_faetures], target, cv)
                score  = cross_validation.cross_val(
                            algorithms,
                            k_folds_dict_data, 
                            metric, 
                            task='regr', 
                            cv=cv, 
                            verbose=0,
                            early_stopping_rounds=early_stopping_rounds)

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
    early_stopping_rounds default 0 or positive int.
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
    
    k_folds_dict_data = splitter.k_folds_split(df[features], target, cv)
    score_with_all_features  = cross_validation.cross_val(
                                        algorithms,
                                        k_folds_dict_data, 
                                        metric, 
                                        task='clsf', 
                                        cv=cv, 
                                        verbose=0,
                                        early_stopping_rounds=early_stopping_rounds)
    if verbose:
        print(f'{score_with_all_features}     General cv_score with all features')
    
    
    while score_with_all_features <= cv_score_of_bad_feature:
        scores_df = pd.DataFrame(index = features, columns=['cv_score'])
        k_folds_dict_data = splitter.k_folds_split(df[features], target, cv)
        score_with_all_features  = cross_validation.cross_val(
                                        algorithms,
                                        k_folds_dict_data, 
                                        metric, 
                                        task='clsf', 
                                        cv=cv, 
                                        verbose=0,
                                        early_stopping_rounds=early_stopping_rounds)
        if verbose:
            print('\n\n')
            print(f'{len(features)} number of features')
            print("{:1.8f}   {:20}  ".format(score_with_all_features, 'BASE cv_score with all features'))
            print('\n\nwithou feature\n')

        for without_feature in features:
            if without_feature != target:
                fit_faetures = features[:]
                fit_faetures.remove(without_feature)
                k_folds_dict_data = splitter.k_folds_split(df[fit_faetures], target, cv)
                score  = cross_validation.cross_val(
                                        algorithms,
                                        k_folds_dict_data, 
                                        metric, 
                                        task='clsf', 
                                        cv=cv, 
                                        verbose=0,
                                        early_stopping_rounds=early_stopping_rounds)
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
