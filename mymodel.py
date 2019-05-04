import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

parse_dates = ['update_date', 'report_date']
df_cust = pd.read_csv('customer_data_ratio20.csv')
df_payment = pd.read_csv('payment_data_ratio20.csv', parse_dates=parse_dates)

df_merge = pd.merge(df_cust, df_payment, how = 'right', left_on='id', right_on='id')


### Data Exploration 
# Set up dependent variable
y = df_merge['label']
y.sample(5)
# proportion for each class
# class 0 = negative class = low risk
# class 1 = positive class = high risk 
ratio_y= np.bincount(y)/len(y)
print('proportion for class 0 (low risk) is {}'.format(ratio_y[0]))
print('proportion for class 1 (high risk) is {}'.format(ratio_y[1]))


X = df_merge.drop(axis=1, columns ='label')
X.sample(5)
X.info()
X.describe().transpose()

# I will drop report_date and update_date for the time being,
# and columns with missing values fea_2, prod_limit and  highest_balance
# I will see later how to deal with this 
X = X.drop(axis=1, columns =['report_date', 'update_date'])
X = X.dropna(axis=1)
# Categorical variables
# looking at the features, I can identify several variables that can be considered categorical
# fea_1, fea_3, fea_5, fea_6, fea_7, fea_9 
Feat_cat = ['fea_1', 'fea_3', 'fea_5', 'fea_6', 'fea_7', 'fea_9' ]
X.shape

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

## Negative class (0) is most frequent
#dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
## Therefore the dummy 'most_frequent' classifier always predicts class 0
#y_dummy_predictions = dummy_majority.predict(X_test)
#print('Dummy majority score = {}'.format(dummy_majority.score(X_test, y_test)))
#
#dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)
#print('Dummy proportion score = {}'.format(dummy_classprop.score(X_test, y_test)))

## Decision tree classifier with random choice of paramters
dt = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_train, y_train)
print('Decision Tree accuracy score for max_depth = 10 is : {}'.format(dt.score(X_test, y_test)))

dt = DecisionTreeClassifier(max_depth=20, random_state=0).fit(X_train, y_train)
print('Decision Tree accuracy score for max_depth = 20 is : {}'.format(dt.score(X_test, y_test)))

dt = DecisionTreeClassifier(max_depth=100, random_state=0).fit(X_train, y_train)
print('Decision Tree accuracy score for max_depth = 100 is : {}'.format(dt.score(X_test, y_test)))

## searching for the best parameters 
param_grid = {'max_depth': np.arange(3,20),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}

## for f1 measure 
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'f1')

grid_tree.fit(X_train, y_train)
best_params = grid_tree.best_params_
dt_f1 = DecisionTreeClassifier()
dt_f1.set_params(**best_params)
dt_f1.fit(X_train, y_train)
tree_predicted = dt_f1.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))

### for recall 
#
#grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'recall')
#
#grid_tree.fit(X_train, y_train)
#best_params = grid_tree.best_params_
#dt_rec = DecisionTreeClassifier()
#dt_rec.set_params(**best_params)
#dt_rec.fit(X_train, y_train)
#tree_predicted = dt_rec.predict(X_test)
#print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
#print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
#print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
#print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))

## dealing with imbalanced classes and impact on model performance 

sm = SMOTE(random_state=589, ratio = 1.0)
X_SMOTE, y_SMOTE = sm.fit_sample(X_train, y_train)
print(len(y_SMOTE))
print(y_SMOTE.sum())
