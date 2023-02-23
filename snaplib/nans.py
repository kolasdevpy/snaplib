import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import counter






def nan_info(df : pd.DataFrame) -> pd.DataFrame:
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




def nan_plot(df : pd.DataFrame) -> plt.figure:
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
    # plt.show()
    return plt




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
    

    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
    if target is not None:
        if type(target) != str:
            raise TypeError('target must be of str type.')
    if type(verbose) != int and type(verbose) != bool:
        raise TypeError('verbose must be of int type or bool.')

    counter_droped_columns = counter.Counter(start=0)

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
        
    for col in df.columns:
        unique_array = np.array(df[col].unique())
        if len(unique_array) == 1:
            df.drop([col], inplace=True, axis=1)
            counter_droped_columns(1)
            if verbose:
                print(f'column "{col}" cnontains 1 unique value - has been dropped')
        elif len(unique_array) == 2 and np.any(pd.isnull(df[col])):
            if verbose:
                print(f'!!! column "{col}" cnontains 1 unique value and np.nan')
    
    if verbose:
        print(f'{counter_droped_columns.counter} columns have been dropped')
        print(f'shape: {df.shape}\n')

    # DROP ROWS with NaN IN TARGET
    if target:
        if verbose:
            print('DROP ROWS with NaN IN TARGET:')
        nan = df[df[target].isnull()]
        indeces_todrop = list(nan.index)
        if verbose:
            print(f'{len(indeces_todrop)} rows have been dropped')
        df = df.drop(df.index[indeces_todrop])
        df.reset_index(drop=True, inplace=True)
    if verbose:
        print(f'shape: {df.shape}\n')
        print(f'\nFinish shape: {df.shape}\n')
    return df






