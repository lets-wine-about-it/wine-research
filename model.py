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

def get_baseline_accuracy(x_trains, y_train):
    '''get baseline accuracy score'''
    
    # assign most common class to baseline
    baseline = y_train.mode()
    
    # compare baseline with y_train class to get most common class
    matches_baseline_prediction = (y_train == 6)
    
    # get mean
    baseline_accuracy = matches_baseline_prediction.mean()
    
    # print baseline accuracy
    print(f"Baseline accuracy: {baseline_accuracy}")


    
def decision_tree_loop(x_trains, x_validates, y_train, y_validate):
    
    ''' This method pulls in a x_train, x_validate, y_train, y_validate dataframes and returns a dataframe which holds the train accuracy score, validate score and the accuracy difference ''' 
    
    # create an empty list to append output
    metrics = []

    for i in range(1,10):
    # create model
        clf = DecisionTreeClassifier(max_depth=i, random_state=42)

        # fit the model to training data
        clf.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = clf.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate = clf.score(x_validates,y_validate)

        output = {'max_depth': i,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df['difference'] = df.train_accuracy - df.validate_accuracy
    return df

def random_forest_tree_loop(x_trains, x_validates, y_train, y_validate):
    
    # create an empty list to append output
    metrics = []
    
    for i in range(1, 25):
    
        # create model
        rf = RandomForestClassifier(min_samples_leaf =i, random_state=42) 

        # fit the model to training data
        rf.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = rf.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate = rf.score(x_validates,y_validate)

        output = {'min_samples_leaf ': i,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df['difference'] = df.train_accuracy - df.validate_accuracy
    return df


def knn_loop(x_trains, x_validates, y_train, y_validate):
    
    # create an empty list to append output
    metrics = []
    
    for i in range(1, 25):

        # create model
        knn = KNeighborsClassifier(n_neighbors=i) 

        # fit the model to training data
        knn.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = knn.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate = knn.score(x_validates,y_validate)

        output = {'n_neighbors': i,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df['difference'] = df.train_accuracy - df.validate_accuracy
    return df


def get_decision_tree(x_trains, x_validates, y_train, y_validate, n):
    '''get decision tree accuracy score on train and validate data'''
    
    # create model
    clf = DecisionTreeClassifier(max_depth = n, random_state=42)

    # fit the model to train data
    clf.fit(x_trains, y_train)

    # compute accuracy
    train_acc = clf.score(x_trains, y_train)
    validate_acc = clf.score(x_validates, y_validate)
    
    # print accuracy score on train
#     print(f'Decision Tree Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
#     print(f'Decsion Tee Accuracy score on validate set: {validate_acc}')
    
    return train_acc, validate_acc


def get_random_forest(x_trains, x_validates, y_train, y_validate, n):
    '''get random forest accuracy score on train and validate data'''
    
    # create model
    rf= RandomForestClassifier(min_samples_leaf = n, random_state=42) 

    # fit the model to train data
    rf.fit(x_trains, y_train)

    # compute accuracy
    train_acc = rf.score(x_trains, y_train)
    validate_acc = rf.score(x_validates, y_validate)
    
    # print accuracy score on train
#     print(f'Random Forest Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
#     print(f'Random Forest score on validate set: {validate_acc}')
    return train_acc, validate_acc


def get_knn(x_trains, x_validates, y_train, y_validate, n):
    ''' get KNN accuracy score on train and validate data'''
    
    # create model
    knn= KNeighborsClassifier(n_neighbors = n) 

    # fit the model to train data
    knn.fit(x_trains, y_train)

    # compute accuracy
    train_acc = knn.score(x_trains, y_train)
    validate_acc = knn.score(x_validates, y_validate)
    
    # print accuracy score on train
#     print(f'KNN Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
#     print(f'KNN Accuracy score on validate set: {validate_acc}')
    
    return train_acc, validate_acc
    
def get_models_accuracy(x_trains, y_train, x_validates, y_validate, train, validate):
    '''takesx_trains, y_train, x_validates, y_validate, train, validate, target
    return dataframe with models and their RMSE values on train and validate data
    '''
    # get accuracy
#     baseline_accuracy = mo.get_baseline_accuracy(x_trains, y_train)
    tree_train_acc, tree_validate_acc= get_decision_tree(x_trains, x_validates, y_train, y_validate, 2)
    random_train_acc, random_validate_acc= get_random_forest(x_trains, x_validates, y_train, y_validate, 22)
    knn_train_acc, knn_validate_acc= get_knn(x_trains, x_validates, y_train, y_validate, 19)
    
    # assing index
    index = ['Decision_Tree(max_depth=2)', 'Random_Forest(min_samples_lead=22)', 'KNN (Neighours=19)']
    
    # create a dataframe
    df = pd.DataFrame({'train_accuracy':[tree_train_acc, random_train_acc, knn_train_acc],
                         'validate_accuracy': [tree_validate_acc, random_validate_acc, knn_validate_acc]},index=index)
    df['difference']= df['train_accuracy']-df['validate_accuracy']
    
    return df


def get_decison_tree_test(x_train, x_test, y_train, y_test,n):
   

        clf = DecisionTreeClassifier(max_depth=n, random_state=42)
        clf.fit(x_train, y_train)
    
       
        validate_acc = clf.score(x_test, y_test)
        
        
        print(validate_acc)
        
        
def get_random_forest_test(x_train, x_test, y_train, y_test,n):
   

        rf= RandomForestClassifier(min_samples_leaf = n, random_state=42) 
        rf.fit(x_train, y_train)
    
       
        validate_acc = rf.score(x_test, y_test)
        
        
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



    
    

    

    
    
    
    
     