import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from typing import Callable


from . import nans
from . import encoder
from . import splitter
from . import cross_validation
from . import recover
from . import fit_pred_bagged
from . import feature_selection_bagged
from . import save_load_algorithms










def inherit_doc(doc : str) -> Callable:
    '''
    Decorator: inheritance outer func.__doc__.
    '''
    def _decorator(func : Callable) -> Callable:
        func.__doc__ = doc
        return func
    return _decorator










class Snaplib:
    '''
    data preprocessing library

    This library has methods for preprocessig data : pandas.DataFrame
        1. nan_info
        2. nan_plot
        3. cleane
        4. recover_data - NaN imputing with ML,
        5. train_test_split_balanced
        6. encode_dataframe
        7. decode_dataframe
        8. k_folds_split

    For one and list of algorithms
        9. cross_val
        10. features_selection_clsf
        11. fit_stacked
        12. save_stack
        13. load_stack
        14. predict_stacked
    '''

    ENCODER = encoder.Encoder()




    @staticmethod
    @inherit_doc(nans.nan_info.__doc__)
    def nan_info(*args, **kwargs):
        return nans.nan_info(*args, **kwargs)


    @staticmethod
    @inherit_doc(nans.nan_plot.__doc__)
    def nan_plot(*args, **kwargs):
        return nans.nan_plot(*args, **kwargs)


    @staticmethod
    @inherit_doc(nans.cleane.__doc__)
    def cleane(*args, **kwargs): 
        return nans.cleane(*args, **kwargs)


    @inherit_doc(encoder.Encoder.encode_dataframe.__doc__)
    def encode_dataframe(self, *args, **kwargs): 
        return self.ENCODER.encode_dataframe(*args, **kwargs)


    @inherit_doc(encoder.Encoder.decode_dataframe.__doc__)
    def decode_dataframe(self,*args, **kwargs): 
        return self.ENCODER.decode_dataframe(*args, **kwargs)


    @staticmethod
    @inherit_doc(splitter.k_folds_split.__doc__)
    def k_folds_split(*args, **kwargs): 
        return splitter.k_folds_split(*args, **kwargs)


    @staticmethod
    @inherit_doc(splitter.train_test_split_balanced.__doc__)
    def train_test_split_balanced(*args, **kwargs):
        return  splitter.train_test_split_balanced(*args, **kwargs)


    @staticmethod
    @inherit_doc(cross_validation.cross_val.__doc__)
    def cross_val(*args, **kwargs):
        return cross_validation.cross_val(*args, **kwargs)


    @staticmethod
    @inherit_doc(recover.recover_data.__doc__)
    def recover_data(*args, **kwargs):
        return recover.recover_data(*args, **kwargs)


    @staticmethod
    @inherit_doc(fit_pred_bagged.fit_stacked.__doc__)
    def fit_stacked(*args, **kwargs):
        return  fit_pred_bagged.fit_stacked(*args, **kwargs)


    @staticmethod
    @inherit_doc(fit_pred_bagged.predict_stacked.__doc__)
    def predict_stacked(*args, **kwargs):
        return  fit_pred_bagged.predict_stacked(*args, **kwargs)


    @staticmethod
    @inherit_doc(feature_selection_bagged.features_selection_regr.__doc__)
    def features_selection_regr(*args, **kwargs):
        return feature_selection_bagged.features_selection_regr(*args, **kwargs)


    @staticmethod
    @inherit_doc(feature_selection_bagged.features_selection_clsf.__doc__)
    def features_selection_clsf(*args, **kwargs):
        return feature_selection_bagged.features_selection_clsf(*args, **kwargs)


    @staticmethod
    @inherit_doc(save_load_algorithms.save_stack.__doc__)
    def save_stack(*args, **kwargs):
        return save_load_algorithms.save_stack(*args, **kwargs)


    @staticmethod
    @inherit_doc(save_load_algorithms.load_stack.__doc__)
    def load_stack(*args, **kwargs):
        return save_load_algorithms.load_stack(*args, **kwargs)




#EOF