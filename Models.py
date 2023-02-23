# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 01:29:48 2023

@author: Yousha
"""

## Importing and prep ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('Cleaned data.csv')
df_outliers = pd.read_csv('Data Outliers.csv')

df.columns
df_outliers.columns

features = ['alcohol','density','chlorides','volatil_acidity',\
            'total_sulfur_dioxide','fixed_acidity','pH','residual_sugar']

X1 = df[features]
X2 = df_outliers[features]
X3 = df.drop('quality',axis=1)
y1 = df['quality']
y2 = df_outliers['quality']

## MODELS ##

# Random Forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=(1))
rf2 = RandomForestClassifier(random_state=(1))

rf.fit(X1,y1)
rf.score(X1,y1) #99.95%

rf2.fit(X2,y2)
rf2.score(X2,y2) #99.97%

# Overtraining?
# Yup

from sklearn.model_selection import train_test_split

X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,stratify = y1,random_state=1,test_size=.2)
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,stratify = y2,random_state=1,test_size=.2)
X3_train,X3_test,y1_train,y1_test = train_test_split(X3,y1,stratify = y1,random_state=1,test_size=.2)

rf.fit(X1_train,y1_train) 
print(rf.score(X1_test,y1_test).round(4)) #70.2%

rf2.fit(X2_train,y2_train)
print(rf2.score(X2_test,y2_test).round(4)*100) #68.47%

rf.fit(X3_train,y1_train)
print(rf.score(X3_test,y1_test).round(4)*100) #70.61%  Best

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rf,X3,y1,cmap=plt.cm.magma)
plt.title('Random Forest Model')
plt.savefig('Random Forest CM.png',dpi=600,bbox_inches='tight')

from sklearn.metrics import classification_report
#classification_report(y2_test, y_pred)

# Stratifying is important

## TUNING ##

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf3 = RandomForestClassifier()
rf_rscv = RandomizedSearchCV(estimator=rf3, param_distributions=random_grid,
                             n_iter = 100, cv = 3, verbose=2, random_state=42, 
                             n_jobs = -1)
rf_rscv.fit(X3_train,y1_train)
rf_rscv.best_params_

rf_rand = RandomForestClassifier(n_estimators= 1000,min_samples_split= 2,
                                 min_samples_leaf= 1,max_features= 'sqrt',
                                 max_depth=20,bootstrap= True) # Best params rscv

rf_rand.fit(X3_train,y1_train)
rf_rand.score(X3_test,y1_test) #70.82% 
y_pred = rf_rand.predict(X3_test)

## Grid Search

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators':[500,1000,1500],
              'min_samples_split':[2,4,6],
              'min_samples_leaf':[1,2,3],
              'max_features':[2,3],
              'max_depth':[30,40,50,60],
              'bootstrap':[True]}
rf4 = RandomForestClassifier()

rf_gscv = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
rf_gscv.fit(X3_train,y1_train)
rf_gscv.best_params_

rf_fin = RandomForestClassifier(n_estimators= 1000,min_samples_split= 2,
                                 min_samples_leaf= 1,max_features=3,
                                 max_depth=30,bootstrap= True) # Best params GridSearch

rf_fin.fit(X3_train,y1_train)
rf_fin.score(X3_test,y1_test) #71.33% # Best






