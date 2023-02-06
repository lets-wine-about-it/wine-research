import pandas as pd  
import numpy as np

# Visual Imports
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

seed =42


def wine_data():
    '''read two csv files separately from local storage and return a dataframe made of combined data of csv files'''
    
    # read data from csv to a dataframe
    red = pd.read_csv('winequality-red.csv')
    white = pd.read_csv('winequality-white.csv')

    # create a new column for a dataframe
    red["type"] = 'red'
    white['type'] = 'white'

    # concate dataframe 
    wines = pd.concat([red, white], axis=0)
    
    # create a dictionary
    name_dict = {'fixed acidity':'fixed_acidity', 'volatile acidity': 'volatile_acidity','citric acid':'citric_acid','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}
    
    # rename columns
    wines.rename(columns = name_dict, inplace = True)
    
    return wines


# def get_bins(df):
#     take
      
#     new_data = { 3:"poor",
#              4:"poor",
#              5:"average",
#              6:"average",
#              7:"good",
#              8:"good",
#              9:"good"}
    
#     # combine this new data with existing DataFrame
#     df["quality_bin"] = df["quality"].map(new_data)
    
#     return df
    
    
def get_dummies(df, columns):
    '''takes in a dataframe and columns names and return  dataframe with encoded vairables'''
    
    df_dummies = pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep='_', drop_first=True)
    
    return df_dummies


def split_data(df,strat):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    df = get_dummies(df,['type'])
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=42,stratify= df[strat])
    
    train, validate = train_test_split(train_validate, test_size=.3, random_state=42,stratify= train_validate[strat] )
    
    return train, validate, test


def train_val_test(df, strat):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # create a dictionary
    new_data = { 3:"poor",
             4:"average",
             5:"average",
             6:"average",
             7:"average",
             8:"good",
             9:"good"}
    
    # combine this new data with existing DataFrame
    df["quality_bin"] = df["quality"].map(new_data)
    
#     df= get_dummies(df)
    
    train, validate, test = split_data(df,strat)
    
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


def scale_data(train, validate, test, columns_to_scale=[], return_scaler=False):
    ''' 
    takes in train, validate, and test data and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    
    # make copies of our original data 
    train_scaled = train.drop(columns='quality_bin').copy()
    validate_scaled = validate.drop(columns='quality_bin').copy()
    test_scaled = test.drop(columns='quality_bin').copy()
    
    # use sacaler
    scaler = MinMaxScaler()
    
    # fit scaler
    scaler.fit(train[columns_to_scale])
    
    # apply the scaler
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                             columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
