#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Classification 
In this dataset, each customer is classified as high or low credit risk according to the set of features and payment history. If label is 1, the customer is in high credit risk. 
Dataset imbalance ratio is 20%.

Data:
payment_data.csv: customer’s card payment history.
	id: customer id
	OVD_t1: number of times overdue type 1
	OVD_t2: number of times overdue type 2
	OVD_t3: number of times overdue type 3
	OVD_sum: total overdue days
	pay_normal: number of times normal payment
	prod_code: credit product code
	prod_limit: credit limit of product
	update_date: account update date
	new_balance: current balance of product
	highest_balance: highest balance in history
	report_date: date of recent payment
customer_data.csv: customer’s demographic data and category attributes which have been encoded. Category features are fea_1, fea_3, fea_5, fea_6, fea_7, fea_9.

Tasks:
•	Explore data to give insights.
•	Build features from existing payment data.
•	Build model to predict high risk customer 
•	Model explanation and evaluation

# ## Synopsis 
# 
# For this Data analysis, we wanto to classify customers as low credit risk (class 0) or as high credit risk (class 1).
# We will proceed in steps:
# - Step 1: We will start with data exploration and cleansing:
#     In this steps we will be handling missing values
# - Step 2: We will build features and target variables
#     We will build the train and test set and propose a method to deal with imbalanced class
# - Step 3: We will explore several models and evaulate their performance
#     We will look at 3 models:
#         1. Decision Tree Classifier
#         2. Random Forest Classifier 
#         3. Gradient Boosting Tree Classifier 
#     We will try to find the best set of parameters with a f1-driven study, using GridSearchCV
# - Step 4: We will compare the model and propose several directions to progress with this analysis 
# 
# ###  Note on performance:
# One main driver in our model evalutation will be to find the right balance between recall and precision. 
# Indeed, we want to control the recall to ensure our model does not classify as high risk, customers who are low risk and we also want to make sure to identify all customers with high credit risk profile. In other words, we want to find the right balance between customer satisfaction and risk for the bank. 

# In[56]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from imblearn.over_sampling import SMOTE


# ## Step 1: Data exploration 

# In[9]:


#loading the data 
parse_dates = ['update_date', 'report_date']
df_cust = pd.read_csv('customer_data_ratio20.csv')
df_payment = pd.read_csv('payment_data_ratio20.csv', parse_dates=parse_dates)
#We map each payment instance to the customer demographic by merging on 'id'
df = pd.merge(df_cust, df_payment, how = 'right', left_on='id', right_on='id')

df.sample(5)


# In[8]:


df.shape


# In[11]:


df.info()


# In[13]:


df.describe().transpose()


# ### Missing values - Data cleaning 
# 1. Some dates are missing for 'report_date' and 'update_date'
#     We will fill the missing values from each column using the other column
# 2. hightest_balance also has a number of missing values that we will fill with 'new_balance' 
# 3. fea_2 missing values will be replaced by the mean of the column 
# 4. missing values for prod_limit will be replaced by 0 (this could mean that 0 credit limit is set) 

# In[16]:


df['report_date']= df['report_date'].fillna(df['update_date'])
df['update_date']= df['report_date'].fillna(df['update_date'])
df['highest_balance']= df['highest_balance'].fillna(df['new_balance'])
df['fea_2']= df['fea_2'].fillna(df['fea_2'].mean())
df['prod_limit']= df['prod_limit'].fillna(0)


# In[17]:


df.info()


# we still have about 24 rows with missing values for report_date, we will remove these rows 

# In[19]:


df = df.dropna(axis=0)
df.info()


# In[20]:


df.shape


# We will work with 8226 instances

# ## Step 2: Building features 

# The data will be separated into 
#     - our target, the column 'label' that has 2 classes: 0 for low risk and 1 for high risk 
#     - features, the other columns of df
# 

# In[61]:


y = df['label']

class_ratio = np.bincount(y)/len(y)
class_ratio


# We can see the class are imbalanced with 17% of high risk and 83% of low risk.
# 
# We will be using SMOTE to 'rebalance' the training set 

# In[69]:


X = df.drop(axis=1, columns = 'label')

#We will use 'update_date' and 'report_date' as categorical  variables in the model
#Set concoder
encoder = LabelEncoder()
X['update_date'] =  encoder.fit_transform(X['update_date'])
X['report_date'] =  encoder.fit_transform(X['report_date'])


# In[70]:



X_train, X_test, y_train, y_test =  train_test_split(X,y)
X_SMOTE, y_SMOTE = X_train, y_train


# In[26]:


#sm = SMOTE(random_state = 0, ratio = 1)
#X_SMOTE, Y_SMOTE = sm.fit_sample(X_train, y_train)
#ratio_y_SMOTE = nb.bincount(y_SMOTE) /len(y_SMOTE)


# ##  Step 3: Building model 

# We will be exploring 3 models:
#     1. Decisition Tree Classifier
#     2. Random Forest Classier
#     3. Gradient Boosting Classifier
#     
# For each of these models, we will use GridSearchCV to search for the best paramaters to outperform the F1 score.
# This metric will enable to control both recall and precision.
# We want our model:
#     1. to have good recall to minimise the risk of classifing clients as high risk when they are not (customer satisfaction) 
#     2. to have a good precision to minimise the risk of classigfing clients as low risk when they are high risk (bank protection)
# 

# ### Decision Tree Classifier 

# In[71]:


# Decision Tree Classififier with max_depth = 10
dt = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_SMOTE, y_SMOTE)
accuracy_score(y_test, dt.predict(X_test))


# In[72]:


# Decision Tree Classififier with max_depth = 20
dt = DecisionTreeClassifier(max_depth=20, random_state=0).fit(X_SMOTE, y_SMOTE)
accuracy_score(y_test, dt.predict(X_test))


# In[75]:


# Search optimal parameters for F1 performance
dt_params = {'max_depth': np.arange(3,20),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}
grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid = dt_params, cv = 5, scoring= 'f1')

grid_dt.fit(X_SMOTE, y_SMOTE)
best_params_dt = grid_dt.best_params_
dt_f1 = DecisionTreeClassifier(random_state = 0)
dt_f1.set_params(**best_params_dt)
dt_f1.fit(X_SMOTE, y_SMOTE)
dt_predicted = dt_f1.predict(X_test)


# In[51]:


print(best_params_dt)
print('Decision tree fitted on balanced sample - Accuracy: {:.2f}'.format(accuracy_score(y_test, dt_predicted)))
print('Decision tree fitted on balanced sample - Precision: {:.2f}'.format(precision_score(y_test, dt_predicted)))
print('Decision tree fitted on balanced sample - Recall: {:.2f}'.format(recall_score(y_test, dt_predicted)))
print('Decision tree fitted on balanced sample - F1: {:.2f}'.format(f1_score(y_test, dt_predicted)))


# We can observe that the recall is quite low and this model is not satisfying at this stage

# ### Random Forest Classifier 

# In[74]:


# Search optimal parameters for F1 performance
rdf_params = {'n_estimators' : [10,15,20],
              'max_depth': np.arange(3,20),
              'criterion' : ['gini','entropy'],
              'max_features': [1,5,10,18]}
grid_rdf = GridSearchCV(RandomForestClassifier(), param_grid = rdf_params, cv = 5, scoring= 'f1')


# In[60]:


grid_rdf.fit(X_SMOTE, y_SMOTE)


# In[ ]:


best_params_rdf = grid_rdf.best_params_
rdf_f1 = RandomForestClassifier(random_state = 0)
rdf_f1.set_params(**best_params_dt)
rdf_f1.fit(X_SMOTE, y_SMOTE)
rdf_predicted = rdf_f1.predict(X_test)


# In[ ]:


print(best_params_rdf)
print('Random Forest fitted on balanced sample - Accuracy: {:.2f}'.format(accuracy_score(y_test, rdf_tree_predicted)))
print('Random Forest fitted on balanced sample - Precision: {:.2f}'.format(precision_score(y_test, rdf_tree_predicted)))
print('Random Forest fitted on balanced sample - Recall: {:.2f}'.format(recall_score(y_test, rdf_tree_predicted)))
print('Random Forest fitted on balanced sample - F1: {:.2f}'.format(f1_score(y_test, rdf_tree_predicted)))


# We can observe much better performance, especially on the recall, as compared to Decision Tree Classification.
# Let's see if the Gradient Boosting Decision Tree will give us even better result

# ### Gradient Boosting Decision Tree Classifier 

# For the Gradient Boosting Tree Classifier, we will use the same parameters as for the Random Forest model, and we will set the learning_rate to 0.1
# 

# In[67]:


gbdt_f1 = GradientBoostingClassifier(random_state = 0, n_estimators = 20, max_depth = 18,
                                     max_features = 5, learning_rate = 0.1)

gbdt_f1.fit(X_SMOTE,y_SMOTE)
gbdt_predict = gbdt_f1.predict(X_test)


# In[ ]:


print('Gradient Boosting Tree fitted on balanced sample - Accuracy: {:.2f}'.format(accuracy_score(y_test, gbdt_tree_predicted)))
print('Gradient Boosting Tree fitted on balanced sample - Precision: {:.2f}'.format(precision_score(y_test, rdf_tree_predicted)))
print('Gradient Boosting Tree fitted  on balanced sample - Recall: {:.2f}'.format(recall_score(y_test, rdf_tree_predicted)))
print('Gradient Boosting Tree fitted  on balanced sample - F1: {:.2f}'.format(f1_score(y_test, rdf_tree_predicted)))


# ## Step 4:  Model explanation and evaluation

# Given the performance for the 3 models, the Gradient Boosting Decision Tree Classifier provides the best balance between recall and precision.
# 
# To improve the model we could:
#     
#     1. Tune the parameters with oGridSearchCV and K-fold cross-validation with different parameters 
#     2. Explore the features more with a correlation heatmap and see the impact of removing certain features to the
#     model performance
#     3. Explore the data more to address outliers 
#     4. Explore other models
#     5. Obtain more data
#     
