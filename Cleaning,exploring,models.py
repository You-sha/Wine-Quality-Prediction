# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:13:03 2023

@author: Shumail
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('winequality-white.csv')
df.columns


## CLEANING ##

df[['fixed_acidity',"volatil_ acidity","citric_acid","residual_sugar",
   "chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density",
   "pH","sulphates","alcohol","quality"]] = df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'].apply(\
                                               lambda x: pd.Series(str(x).split(";"))
                                               )

df.drop('fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"', axis=1, inplace=True)

df = df.astype('float')


## VISUALIZING DATA ##

plt.figure(figsize=(10, 10))
correlations = pd.DataFrame(df.corr()["quality"].sort_values(ascending=False))
sns.heatmap(correlations,annot=True,square=True,linewidth=2,cbar_kws={'shrink':0.96})
plt.title('Correlation with Quality',fontsize=18, pad=30)
plt.savefig('Heatmap.png',dpi=600)


## MODELS ##

X = df.drop('quality',axis=1)
y = df['quality']

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
model = LinearRegression()

X_train,X_test,Y_train,Y_test = train_test_split(X,y,random_state=1)

# Linear Regression #
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
model.score(X_test,Y_test).round(2)  # 0.29

from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(solver='liblinear')

# Logistic Regression #
model2.fit(X_train,Y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,Y_test).round(2)  # 0.51


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import GridSearchCV

# KNeighbors Classifier #
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(2,21)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X_train,Y_train)

knn_fin = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_fin.fit(X,y)
knn_gscv.best_params_

y_pred3 = knn_fin.predict(X_test)
knn_fin.score(X,y).round(2)  # 0.82

## MODEL EVALUATION AND PLOTS ##

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

plot_confusion_matrix(knn_fin, X_test, Y_test, cmap=plt.cm.Blues)
plt.xlabel('Predicted Quality')
plt.ylabel('Actual quality')
plt.suptitle('Predicting Wine Quality',fontsize=13)
plt.title('KNN',fontsize=10)
plt.savefig('cm',dpi=600)

plot_confusion_matrix(model2, X_test, Y_test, cmap=plt.cm.Blues)
plt.xlabel('Predicted Quality')
plt.ylabel('Actual quality')
plt.suptitle('Predicting Wine Quality',fontsize=13)
plt.title('Logistic Regression',)
plt.savefig('logistic_cm',dpi=600)

mean_squared_error(Y_test,y_pred3).round(2)
mean_absolute_error(Y_test,y_pred).round(2)
round(np.var(y),2)






































