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



# Use cases     
      
from snaplib.snaplib import Snaplib     
sl = Snaplib()    
       
     
      
missing_info_df = sl.nan_info(df)     
      
sl.nan_plot(df)      
     
df = sl.cleane(df, target_name, verbose=True)     
      
train_X, test_X, train_y, test_y = sl.train_test_split_balanced(df, target_name, test_size=0.2, random_state=0, research_iter=0)     
      
df = sl.recover_data(df, verbose=True, stacking=True)    

df = sl.encode_dataframe(df)     

df = sl.decode_dataframe(df)     

k_fold_dict_data = sl.k_folds_split(df, target_name_str, k)      

y_hat = sl.fit_predict_stacked(algorithms_list, X_train, y_train, X_pred, y_test=None, task='clsf', verbose=0)     

score = sl.cross_val(algorithms_list, k_fold_dict_data, metric, task, cv, verbose=0)      