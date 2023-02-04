import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import acquire as ac
import prepare as pr

from scipy import stats

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

import warnings
warnings.filterwarnings('ignore')
seed = 42




def cluster_data(train, validate,test, k, cluster_col_name = 'cluster'):
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train)
    
    train_clusters = kmeans.predict(train)
    validate_clusters = kmeans.predict(validate)
    test_clusters = kmeans.predict(test)
    
 
        
    return train_clusters, validate_clusters, test_clusters
        