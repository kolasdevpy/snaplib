import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class snaplib:
    '''preprocessing library'''
    

    def nan_plot(df):
        plt.figure(figsize=(int(len(df.columns)/4) if len(df.columns)>30 else 10, 10))
        plt.pcolor(df.isnull(), cmap='Blues_r')
        plt.yticks([int(el*(len(df)/10)) for el in range(0, 10)])
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=80)
        plt.show()
        return plt
        

    def cleane(df, target=None, verbose=True):
        '''drop_duplicates, drop rows with nan in target, drop column with 1 unique value'''
        
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
            snaplib.nan_plot(df)
        return df