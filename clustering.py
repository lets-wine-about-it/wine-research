
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

seed = 42


def vis_get_elbow(df, x_value, y_value):
    '''takes a dataframe, get inertia for n from 1 to 10 and return a dataframe with inertia'''
    
    df= df[[x_value, y_value]]
    # create an empty list to hold inertia
    inertia = []
    
    for n in range(1, 11):
        
        Kmeans = KMeans(n_clusters=n, random_state=seed)
        
        Kmeans.fit(df)
        
        inertia.append(Kmeans.inertia_)
        
    df= pd.DataFrame({'inertia': inertia, 'cluster': list(range(1,11))})
    
    sns.relplot(data=df, x='cluster', y='inertia', kind='line')
    plt.show()

    
def vis_make_cluster(df, x_value, y_value, n, seed = 42):
    '''takes a dataframe, number of cluster and return a list of clusters'''
    
    cluster_name = x_value +'_' + y_value +'_cluster'
    
    df_1= df.copy()[[x_value, y_value]]
    
    
    Kmeans = KMeans(n_clusters=n, random_state=42)

    Kmeans.fit(df_1)
    
    df_1['clusters']=  Kmeans.predict(df_1)
    df[cluster_name] = df_1['clusters']
    
    sns.relplot(data=df_1, x=df_1[x_value], y=df_1[y_value], hue='clusters')
    plt.show()
    return df
    
