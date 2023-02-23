import pandas as pd
import numpy as np





def fit_stacked(algorithms_list : list, 
                X_train : pd.DataFrame, 
                y_train : pd.Series  or np.ndarray, 
                X_val : pd.DataFrame = None, 
                y_val : pd.Series  or np.ndarray = None, 
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
    if type(algorithms_list) != list:
        raise TypeError('algorithms_list must be of list type.')
    if len(algorithms_list) == 0:
        raise ValueError('algorithms_list is empty.')
    
    if not isinstance(X_train, pd.core.frame.DataFrame):
        raise TypeError('The X__train must be a pandas.core.frame.DataFrame instance.')
    if not isinstance(y_train, pd.core.frame.Series) and not isinstance(y_train, np.ndarray):
        raise TypeError('The y__train must be a pandas.core.frame.Series instance or numpy.ndarray.')

    if X_val is not None:
        if not isinstance(X_val, pd.core.frame.DataFrame):
            raise TypeError('The X__val must be a pandas.core.frame.DataFrame instance.')
    if y_val is not None:
        if not isinstance(y_val, pd.core.frame.Series) and not isinstance(y_val, np.ndarray):
            raise TypeError('The y__val must be a pandas.core.frame.Series instance or numpy.ndarray.')

    if type(verbose) != int and type(verbose) != bool:
        raise TypeError('verbose must be of int type or bool.')
    if type(early_stopping_rounds) != int:
        raise TypeError('early_stopping_rounds must be of int type.')


    for alg in algorithms_list:
        alg_name = alg.__class__.__name__[:3]
        if X_val is not None and \
            y_val is not None and \
            alg_name in ['LGB', 'XGB', 'Cat'] and \
            early_stopping_rounds:

            alg.fit(X_train, 
                    y_train, 
                    eval_set=[(X_val, y_val)], 
                    early_stopping_rounds=early_stopping_rounds, 
                    verbose=verbose)
        else:
            alg.fit(X_train, y_train)

    return algorithms_list











def predict_stacked(algorithms_list : list, 
                    X_pred  : pd.DataFrame, 
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



