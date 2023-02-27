<h1>snaplib</h1>
<h2>A simple data preprocessing tools.</h2>

____________________________________________

# user guide

### Kaggle Notebook

### Classification    
<https://www.kaggle.com/code/artyomkolas/titanic-snaplib-classification/notebook>


### Regression    
<https://www.kaggle.com/code/artyomkolas/housing-prices-with-snaplib/notebook>


____________________________________________

# PyPi

```python
!pip install snaplib
from snaplib.snaplib import Snaplib
sl = Snaplib()
```
       
     
      
1. sl.nan_info     
2. sl.nan_plot     
3. sl.cleane     
4. sl.recover_data - NaN imputing with ML     
5. sl.train_test_split_balanced     
6. sl.encode_dataframe     
7. sl.decode_dataframe     
8. sl.k_folds_split     
### For one and list of algorithms with bagging
9. sl.cross_val    
10. sl.features_selection_regr
11. sl.features_selection_clsf     
12. sl.fit_stacked     
13. sl.save_stack     
14. sl.load_stack     
15. sl.predict_stacked      



# __doc__

```python
print(sl.recover_data.__doc__)
```

Imputing of missing values (np.nan) in tabular data, not TimeSeries.      
      
Use case:
df = Snaplib().recover_data(df, device="cpu", verbose=True)      
device must be "cpu" or "gpu". Sometime small datasets work faster with cpu.      
verbose = True algorithm runs cross validation tests and prints results of tests for decision making.      
discrete_columns = ['col_name_1', 'col_name_2', 'col_name_3', 'etc']      

TESTS on <https://www.kaggle.com/code/artyomkolas/nan-prediction-in-progress/notebook>      