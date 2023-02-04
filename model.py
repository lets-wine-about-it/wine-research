import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import acquire as ac
import prepare as pr
import explore as ex


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

a = .05


def get_tree(x_train, x_validate, y_train, y_validate):
    
    ''' This method pulls in a x_train, x_validate, y_train, y_validate dataframes and returns a dataframe which holds the train accuracy score, validate score and the accuracy difference ''' 
    
    
    results = []
    for md in range(1, 11):
        clf = DecisionTreeClassifier(max_depth=md, random_state=42)
        clf.fit(x_train, y_train)
    
        train_acc = clf.score(x_train, y_train)
        validate_acc = clf.score(x_validate, y_validate)
        accuracy_diff = train_acc - validate_acc
        
        results.append([md, train_acc, validate_acc, accuracy_diff])
        
    results_df = pd.DataFrame(results, columns=['Max Depth', 'Train Accuracy', 'Validate Accuracy', 'Accuracy Difference'])
    return results_df

def get_forest(x_train, x_validate, y_train, y_validate):
    results = []
    for md in range(1, 11):
            rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=md, 
                            random_state=42)

        
            rf.fit(x_train, y_train)
    
            train_acc = rf.score(x_train, y_train)
            validate_acc = rf.score(x_validate, y_validate)
            accuracy_diff = train_acc - validate_acc
        
            results.append([md, train_acc, validate_acc, accuracy_diff])
        
    results_df = pd.DataFrame(results, columns=['Max Depth', 'Train Accuracy', 'Validate Accuracy', 'Accuracy Difference'])
    return results_df


def get_knn(x_train, x_validate, y_train, y_validate):
    results = []
    for n in range(1, 100):
            knn = KNeighborsClassifier(n_neighbors=n, weights='uniform')
        
            knn.fit(x_train, y_train)
    
            train_acc = knn.score(x_train, y_train)
            validate_acc = knn.score(x_validate, y_validate)
            accuracy_diff = train_acc - validate_acc
        
            results.append([n, train_acc, validate_acc, accuracy_diff])
        
    results_df = pd.DataFrame(results, columns=['Max Depth', 'Train Accuracy', 'Validate Accuracy', 'Accuracy Difference'])
    return results_df



def get_tree_test(x_train, x_test, y_train, y_test,md):
   

        clf = DecisionTreeClassifier(max_depth=md, random_state=42)
        clf.fit(x_train, y_train)
    
       
        validate_acc = clf.score(x_test, y_test)
        
        
        print(validate_acc)
     
        
def pearson_test(df, feat1,feat2):
    
    # running the test
    r, p = stats.pearsonr(df[feat1], df[feat2])
    
    print(f'p is {p:.10f}, {r}') 
   

    if p < .05:
        print('The pearson r test shows that there is a signficant relationship.')
    else: 
        print('The relationship is not significant')
        
        
def get_heatmap(df):
    
    # creates a correlation matrix
    wine_heat = df.corr()
    
    # create a heatmap
    sns.heatmap(data = wine_heat,annot= True)
    
def alc_t_test(train):
    
    # creating the average alcohol rating
    alc_mean = train['alcohol'].mean() 
    
    # creating subsets based on the average
    above_alc = train[train['alcohol'] > alc_mean].quality
    below_alc = train[train['alcohol'] < alc_mean].quality
    
    # creating a ttest
    t, p = stats.ttest_ind(above_alc, below_alc, equal_var=False)

    print(f'p = {p}, t = {t}')
    
    print("Reject $H_{0}$? ", p < a)
    
def den_t_test(train):
    
    # creating the density average
    den_mean = train['density'].mean()
    
    # creating subsets bases on the average
    above_den = train[train['density'] > den_mean].quality
    below_den = train[train['density'] < den_mean].quality
    
    # creating a t test
    t, p = stats.ttest_ind(above_den, below_den, equal_var=False)
    
    print(f'p = {p}, t = {t}')
    
    print("Reject $H_{0}$? ", p < a)
    
def vol_t_test(train):
    
    # creating the density average
    vol_mean = train['volatile_acidity'].mean()
    
    # creating subsets bases on the average
    above_vol = train[train['volatile_acidity'] > vol_mean].quality
    below_vol = train[train['volatile_acidity'] < vol_mean].quality
    
    # creating a t test
    t, p = stats.ttest_ind(above_vol, below_vol, equal_var=False)
    
    print(f'p = {p}, t = {t}')
    
    print("Reject $H_{0}$? ", p < a)
    
def alc_den_clusters(x_trains,x_validates,x_tests):
    
    # creating the features to cluster on
    talc = x_trains[['alcohol','density']]
    valc = x_validates[['alcohol','density']]
    tealc = x_tests[['alcohol','density']]
    
    # adding the clusters to the original datasets
    x_trains['alc_den'],x_validates['alc_den'],x_tests['alc_den'] = ex.cluster_data(talc,valc,tealc,3,cluster_col_name= 'alc_den')
    
    # plotting the train clusters
    sns.scatterplot(data = x_trains, x = 'alcohol', y = 'density', hue = 'alc_den')
    
    return x_trains, x_validates, x_tests


def den_res_clusters(x_trains,x_validates,x_tests):
    
    # creating the features to cluster on
    tres = x_trains[['residual_sugar','density']]
    vres = x_validates[['residual_sugar','density']]
    teres = x_tests[['residual_sugar','density']] 
    
    # adding the clusters to the original datasets
    x_trains['res_den'],x_validates['res_den'],x_tests['res_den'] = ex.cluster_data(tres,vres,teres,4,cluster_col_name= 'res_den')
    
    # plotting the train clusters
    sns.scatterplot(data = x_trains, x = 'residual_sugar', y = 'density', hue = 'alc_den')    
    return x_trains, x_validates, x_tests

def alc_vol_clusters(x_trains,x_validates,x_tests):
    
    # creating the features to cluster on
    tavol = x_trains[['alcohol','volatile_acidity']]
    vavol = x_validates[['alcohol','volatile_acidity']]
    teavol = x_tests[['alcohol','volatile_acidity']] 
    
    # adding the clusters to the original datasets
    x_trains['alc_vol'],x_validates['alc_vol'],x_tests['alc_vol'] = ex.cluster_data(tavol,vavol,teavol,4,cluster_col_name= 'alc_vol')
    
    # plotting the train clusters
    sns.scatterplot(data = x_trains, x = 'alcohol', y = 'volatile_acidity', hue = 'alc_den')    
    return x_trains, x_validates, x_tests



    
    

    

    
    
    
    
     