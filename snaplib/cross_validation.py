from termcolor import colored
from itertools import chain, combinations
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap

from typing import Callable

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix


from . import counter






def fit_predict_stacked(algorithms_list : list, 
                        X_train : pd.DataFrame, 
                        y_train : pd.Series, 
                        X_test : pd.DataFrame, 
                        y_test : pd.Series = None, 
                        task : str = 'clsf', 
                        verbose : int or bool = 0,
                        early_stopping_rounds : int = 0, 
                        ) -> pd.DataFrame:
    
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
                                verbose= 0, 1,
                                early_stopping_rounds=0 or positve int
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
    if type(early_stopping_rounds) != int:
        raise TypeError('early_stopping_rounds must be of int type.')

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

        if y_test is not None and \
            alg_name in ['LGB', 'XGB', 'Cat'] and \
            early_stopping_rounds:

            model = alg(**params)
            model.fit(X_train, 
                        y_train, 
                        eval_set=[(X_test, y_test)], 
                        early_stopping_rounds=early_stopping_rounds, 
                        verbose=False)
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










def get_score_clsf(args):
    print(f'\r{counter_cv(cpu_num)} / {num_combinations} processed', end='', flush=True)
    algs_list = list(args[0])
    metric = args[1]
    y_hat_set = all_prediction_df.loc[:, algs_list].mode(axis=1)[0].astype('int64')
    score_set = metric(all_prediction_df.loc[:, ['Y_TEST']], y_hat_set)
    return score_set, algs_list


def get_score_regr(args):
    print(f'\r{counter_cv(cpu_num)} / {num_combinations} processed', end='', flush=True)
    algs_list = list(args[0])
    metric = args[1]
    y_hat_set = all_prediction_df.loc[:, algs_list].mean(axis=1)
    score_set = metric(all_prediction_df.loc[:, ['Y_TEST']], y_hat_set)
    return score_set, algs_list




def cross_val(  algorithms : list, 
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
    if type(early_stopping_rounds) != int:
        raise TypeError('early_stopping_rounds must be of int type.')


    global num_combinations
    global all_prediction_df
    global counter_cv
    global cpu_num
    
    
    results=[]    
    all_prediction_df = pd.DataFrame()
    counter_cv = counter.Counter(start=0)
    research_best_score=dict()
    cpu_num = cpu_count()
    num_combinations = 0
    
    
    def powerset(names_ls):
        return chain.from_iterable(combinations(names_ls, r) for r in range(len(names_ls)+1))



    
    for k in range(0, cv):
        pred_frame = fit_predict_stacked(algorithms, 
                                         k_fold_dict['train_X'][k], 
                                         k_fold_dict['train_y'][k], 
                                         k_fold_dict['test_X'][k], 
                                         k_fold_dict['test_y'][k], 
                                         task,
                                         verbose, 
                                         early_stopping_rounds=early_stopping_rounds,
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
                    results_pool.append(p.map(get_score_clsf, params))
            elif task=='regr':
                with Pool(cpu_num) as p:
                    results_pool.append(p.map(get_score_regr, params))

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
    del counter_cv
    del cpu_num

    return score
    
