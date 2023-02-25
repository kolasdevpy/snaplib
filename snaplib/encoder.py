import pandas as pd



class Encoder:

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






    



    def encode_dataframe(self, 
                         df_in : pd.DataFrame, 
                         ) -> pd.DataFrame: 

        '''
        encode a dataframe
        Use case:
        df = Snaplib().encode_dataframe(df)
        '''

        if not isinstance(df_in, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')

        df = df_in.copy()


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




    def encode_column(self, df, column):
        '''
        encode one column in pd.DataFrame.
        '''
        self.encoder_pool[column] = {}
        self.decoder_pool[column] = {}
        not_nan_index=df[df[column].notnull()].index
        values_set = list(set(list(df.loc[not_nan_index, column])))   # sorted?
        value = 0.0
        for el in values_set:
            self.encoder_pool[column][el] = value
            self.decoder_pool[column][value] = el
            value += 1
        df[column] = df[column].map(self.encoder_pool[column])
        df[column] = df[column].astype('float64')
        return df










    def decode_dataframe(self, 
                         df_in : pd.DataFrame, 
                         ) -> pd.DataFrame: 

        '''
        encode a dataframe
        Use case:
        df = Snaplib().decode_dataframe(df)
        '''

        if not isinstance(df_in, pd.core.frame.DataFrame):
            raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')

        df = df_in.copy()

        for col in self.encoded_columns:
            df[col] = self.decode_column(df[[col]], col)
        return df
    



    def decode_column(self, df, column):
        '''
        decode one column in pd.DataFrame.
        '''
        df[column] = df[column].map(self.decoder_pool[column])
        df[column] = df[column].astype('object')
        return df
    

    

