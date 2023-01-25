<h1>snaplib</h1>
<h2>A simple data preprocessing tools.</h2>

____________________________________________

# user guide

### Kaggle Notebook

### Classification    
<https://www.kaggle.com/code/artyomkolas/titanic-snaplib-classification/notebook>


### Regression    
<https://www.kaggle.com/artyomkolas/snaplib-user-guide/notebook>


____________________________________________

# PyPi

!pip install snaplib           
from snaplib.snaplib import Snaplib     
sl = Snaplib()    
       
     
      
1. sl.nan_info     
2. sl.nan_plot     
3. sl.cleane     
4. sl.recover_data - NaN imputing with ML     
5. sl.train_test_split_balanced     
6. sl.encode_dataframe     
7. sl.decode_dataframe     
8. sl.k_folds_split     
### For one and list of algorithms    
9. sl.cross_val    
10. sl.features_selection_clsf     
11. sl.fit_stacked     
12. sl.save_stack     
13. sl.load_stack     
14. sl.predict_stacked      