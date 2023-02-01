# Essential Imports

import pandas as pd  
import numpy as np

# Visual Imports
import matplotlib.pyplot as plt
import seaborn as sns


def wine_data():
    # pulling data from csv
    red = pd.read_csv('winequality-red.csv')
    
    white = pd.read_csv('winequality-white.csv')

    # creating a classifying column per column
    red["type"] = 'red'
    
    white['type'] = 'white'

    wines = pd.concat([red, white], axis=0)

    name_dict = {'fixed acidity':'fixed_acidity', 'volatile acidity': 'volatile_acidity','citric acid':'citric_acid',
             'residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}

    wines.rename(columns = name_dict,
             inplace = True)

    wines['free_sulfur_dioxide'] = wines.free_sulfur_dioxide.astype(int)

    wines['total_sulfur_dioxide'] = wines.total_sulfur_dioxide.astype(int)
    
    return wines