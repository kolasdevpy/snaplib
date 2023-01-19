<h1>snaplib</h1>
<h2>A simple data preprocessing tools.</h2>

____________________________________________

# user guide

**[Kaggle Notebook](https://www.kaggle.com/code/artyomkolas/titanic-snaplib-classification/settings?scriptVersionId=116790545)**   
<https://www.kaggle.com/code/artyomkolas/titanic-snaplib-classification/settings?scriptVersionId=116790545>


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