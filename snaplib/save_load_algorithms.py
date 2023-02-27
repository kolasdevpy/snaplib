import pickle
from os import makedirs





def save_stack( algorithms_list : list, 
                directory : str = '', 
                ) -> None:
    ''' 
    Save method for all in list of algorithms in directory.
    Return list of file names
    
    Use case:
    file_names = save_stack(algorithms_list, directory='')
    
    algorithms_list = list of algorithms [LGBMClassifier(), XGBClassifier(), CatBoostClassifier()].
    directory:
    "" - save files to current working directory
    or
    save files to "/some/directory/not/exist/"
    '''

    if type(algorithms_list) != list:
        raise TypeError('algorithms_list must be of list type.')
    if len(algorithms_list) == 0:
        raise ValueError('algorithms_list is empty.')
    if type(directory) != str:
        raise ValueError('directory must be of str type.')

    names = []
    if directory:
        directory = (directory[:-1] if directory[-1] == '/' else directory) + '/'
    makedirs(directory, exist_ok=True)
    
    for alg in algorithms_list:
        filename = alg.__class__.__name__ + '.sav'
        if filename in names:
            filename =  alg.__class__.__name__  + '_another' + '.sav'
        path = directory + filename
        pickle.dump(alg, open(path, 'wb'))
        names.append(filename)
    return names










def load_stack( names_list : list, 
                directory : str = '',
                ) -> None:
    ''' 
    Load method for file names in list of names in directory.
    Return list of algorithms.
    
    Use case:
    algorithms = load_stack(names_list, directory='')
    
    names_list is the list of names like ['LGBMClassifier.sav', 'XGBClassifier.sav', 'CatBoostClassifier.sav']
    directory:
    "" - read files from current working directory
    or
    read files from "/some/directory/with/models/"
    '''

    if type(names_list) != list:
        raise TypeError('names_list must be of list type.')
    if len(names_list) == 0:
        raise ValueError('algorithms_list is empty.')
    if type(directory) != str:
        raise ValueError('directory must be of str type.')


    algorithms_list = []

    if directory:
        directory = (directory[:-1] if directory[-1] == '/' else directory) + '/'
    
    for alg_name in names_list:
        algorithms_list.append(pickle.load(open(directory + alg_name, 'rb')))
    return algorithms_list