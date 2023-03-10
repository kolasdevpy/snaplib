import pandas as pd




def dummied(df : pd.DataFrame, 
            columns : list[str], 
            ) -> pd.DataFrame: 
    '''
    dummied the passed columns
    Use case:
    df = Snaplib().dummied(df, ["col_1", "col_2", "col_3"])

    Example:
    sl = Snaplib()
    columns = ["Name"]
    df = sl.dummied(df, columns)

    |   | Age | Name |          |   | Age | Name__Nick | Name__Adam | 
    ------------------          ------------------------------------- 
    | 0 |  12 | Nick |          | 0 |  12 |      1     |      0     | 
    | 1 |  27 | Adam |    =>    | 1 |  27 |      0     |      1     | 
    | 2 |  39 | Nick |          | 2 |  39 |      1     |      0     | 
    | 3 |  20 | Adam |          | 3 |  20 |      0     |      1     | 

    '''
    

    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError('The df must be a pandas.core.frame.DataFrame instance.')
    
    if type(columns) != list:
        raise TypeError('columns must be of list type.')
    if len(columns) == 0:
        raise ValueError('columns is empty.')
    for col in columns:
        if type(col) != str:
            raise TypeError('column_name in columns must be of str type.')
        

    for column in columns:
        df[column] = df[column].astype('str')
        dum_df = pd.get_dummies(df[column], prefix=column + '_')
        df = pd.concat([df, dum_df], axis=1)
        df.drop([column], inplace=True, axis=1)
    return df