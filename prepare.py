import pandas as pd  
import numpy as np

# Visual Imports
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer


seed =42


def get_bins(df):
      
    new_data = { 3:"poor",
             4:"poor",
             5:"average",
             6:"average",
             7:"good",
             8:"good",
             9:"good"}
    
    # combine this new data with existing DataFrame
    df["quality_bin"] = df["quality"].map(new_data)
    
    return df
    
    

def get_dummies(df, columns=None, drop_first=True):
    
    dummies = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    
    return dummies


def split_data(df,strat):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=42,stratify= df[strat])
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=42,stratify= train_validate[strat] )
    
    print(train.shape , validate.shape, test.shape)

          
    return train, validate, test


def x_and_y(train,validate,test,target):
    
    """
    splits train, validate, and target into x and y versions
    """

    x_train = train.drop(columns= target)
    y_train = train[target]

    x_validate = validate.drop(columns= target)
    y_validate = validate[target]

    x_test = test.drop(columns= target)
    y_test = test[target]

    return x_train, y_train, x_validate, y_validate, x_test, y_test


def scaled_data(x_train,x_validate,x_test,num_cols,return_scaler = False):

    ''' a function to scale my data appropriately ''' 
    
    # intializing scaler
    scaler = MinMaxScaler()
    
    # fit scaler
    scaler.fit(x_train[num_cols])
    
    # creating new scaled dataframes
    x_train_s = scaler.transform(x_train[num_cols])
    x_validate_s = scaler.transform(x_validate[num_cols])
    x_test_s = scaler.transform(x_test[num_cols])

    # making a copy bof train to hold scaled version
    x_train_scaled = x_train.copy()
    x_validate_scaled = x_validate.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[num_cols] = x_train_s
    x_validate_scaled[num_cols] = x_validate_s
    x_test_scaled[num_cols] = x_test_s

    if return_scaler:
        return scaler, x_train_scaled, x_validate_scaled, x_test_scaled
    else:
        return x_train_scaled, x_validate_scaled, x_test_scaled
    


