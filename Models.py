# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 01:29:48 2023

@author: Yousha
"""

## Importing and prep ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('Cleaned data.csv')
df_outliers = pd.read_csv('Data Outliers3.csv')

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

X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,random_state=1,test_size=.2)
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,random_state=1,test_size=.2)
X3_train,X3_test,y1_train,y1_test = train_test_split(X3,y1,random_state=1,test_size=.2)

rf.fit(X1_train,y1_train) 
print(rf.score(X1_test,y1_test).round(4)) #66.63%

rf2.fit(X2_train,y2_train)
print(rf2.score(X2_test,y2_test).round(4)) #71.43%

rf.fit(X3_train,y1_train)
print(rf.score(X3_test,y1_test).round(4)) #69.59%


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')

lr.fit(X1_train,y1_train) 
print(lr.score(X1_test,y1_test).round(4)) # 50.41%

lr.fit(X2_train,y2_train)
print(lr.score(X2_test,y2_test).round(4)) # 55.96%

lr.fit(X3_train,y1_train)
print(lr.score(X3_test,y1_test).round(4)) # 51.43%

# Gradient boost
from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingClassifier(random_state = 1)
  
gbr.fit(X1_train,y1_train) 
print(gbr.score(X1_test,y1_test).round(4)) # 56.33%

gbr.fit(X2_train,y2_train)
print(gbr.score(X2_test,y2_test).round(4)) # 60.45%

gbr.fit(X3_train,y1_train)
print(gbr.score(X3_test,y1_test).round(4)) # 57.353%

## SCALING AND STANDARDIZATION ##
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
trans = MinMaxScaler()
model = RandomForestClassifier()

X = X1.astype('float32')
y1_scaled = LabelEncoder().fit_transform(y1.astype('str'))

pipeline = Pipeline(steps=[('t', trans), ('m', model)])

# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y1_scaled, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report pipeline performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores))) #Accuracy: 0.692 (0.023)

## MODELS ## 




