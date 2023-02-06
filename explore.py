import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

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


def get_correlation(df):
    '''takes in a dataframe and print correlation of qulity with other features'''
    
    correlation =df.corr()
    
    print(correlation['quality'].sort_values(ascending = False),'\n')
    

def cluster_data(train, validate,test, k, cluster_col_name = 'cluster'):
    
    kmeans = KMeans(n_clusters=k)
    
    kmeans.fit(train)
    
    train_clusters = kmeans.predict(train)
    validate_clusters = kmeans.predict(validate)
    test_clusters = kmeans.predict(test)
       
    return train_clusters, validate_clusters, test_clusters


def vis_get_elbow(df, x_value, y_value):
    '''takes a dataframe, get inertia for n from 1 to 10, create a dataframe with inertia and clusters 
    plot inertia vs clusters'''
    
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
    
    sns.relplot(data=df_1, x=df_1[x_value], y=df_1[y_value], hue='clusters')
    plt.show()

    
def viz_barplot(df, feature_1, feature_2):
    '''takes in a dataframe, features and plot boxplot to show relation of features'''
    
    ax = sns.barplot(data=df, x=feature_1, y=feature_2)
    plt.title(f'Relation of {feature_1} and {feature_2}')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
    
def viz_boxplot(df, feature_1, feature_2):
    '''takes in a dataframe, features and plot barplot to show relation of features'''
    
    ax = sns.boxplot(data=df, x=feature_1, y=feature_2)
    plt.title(f'Relation of {feature_1} and {feature_2}')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
    
def viz_lmplot(df, feature_1, feature_2):
    '''takes in a dataframe, features and plot lmplot to show relation of features'''
    
    ax = sns.lmplot(data=df, x=feature_1, y=feature_2, hue='quality_bin')
    plt.title(f'Relation of {feature_1} and {feature_2}')
    plt.ylim(.988, 1.005)
    plt.xlim(0.0, .2)
#     ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
def viz_jointplot(df, feature_1, feature_2):
    '''takes in a dataframe, features and plot barplot to show relation of features'''

    ax= sns.jointplot(data=df, x=feature_1, y=feature_2, hue='quality_bin')
    plt.title(f'Relation of {feature_1} and {feature_2}')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
    
def pearson_test(df, feat1,feat2):
    '''take in a dataframe and two features, run pearsonr test to show result'''
    
    # run the test
    r, p = stats.pearsonr(df[feat1], df[feat2])
    
    print(f'p is {p:.10f}, {r}') 
   
    if p < .05:
        print('The pearson r test shows that there is a signficant relationship.')
    else: 
        print('The relationship is not significant')
        
              
def viz_histplot(df, colx):
    '''takes in a dataframe, features and plot histplot'''

    good= df[df['quality_bin']=='good']
    average= df[df['quality_bin']=='average']
    poor= df[df['quality_bin']=='poor']
    
    ax = sns.histplot(data=good, alpha=0.5, x=colx,label='good')
    ax = sns.histplot(data=average, alpha=0.2, x=colx,label='average')
    ax = sns.histplot(data=poor, alpha=0.5, x=colx,label='poor')
    plt.legend()
    plt.title('Quality of wine with respect to density')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()