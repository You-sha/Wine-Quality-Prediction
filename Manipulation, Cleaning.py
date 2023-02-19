# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:33:10 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned data.csv')
df.columns

plt.boxplot(
    df[['fixed_acidity', 'volatil_acidity', 'citric_acid', 
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 
        'alcohol', 'quality']])

df_outliers = df[
    ((df.total_sulfur_dioxide < df.total_sulfur_dioxide.quantile(.995))
    &
     (df.total_sulfur_dioxide > df.total_sulfur_dioxide.quantile(.005)))
    &
    ((df.pH < df.pH.quantile(.988))
    &
     (df.pH > df.pH.quantile(.05)))
    &
    ((df.fixed_acidity < df.fixed_acidity.quantile(.995))
     &
     (df.fixed_acidity > df.fixed_acidity.quantile(.005)))
    &
    ((df.residual_sugar < df.residual_sugar.quantile(.995))
     &
     df.residual_sugar > df.residual_sugar.quantile(.005))
    &
    ((df.chlorides < df.chlorides.quantile(.995))
     &
     df.chlorides > df.chlorides.quantile(.005))
    &
    ((df.density < df.density.quantile(.995))
     &
     df.density > df.density.quantile(.005))
    &
    ((df.volatil_acidity < df.volatil_acidity.quantile(.96))
     &
     df.volatil_acidity > df.volatil_acidity.quantile(.005))]

df_outliers2 = df[
    ((df.pH < df.pH.quantile(.995))
    &
     (df.pH > df.pH.quantile(.005)))
    ]


df_outliers.plot.box()
df_outliers2.plot.box()
plt.boxplot(df_outliers['pH'])
df_outliers['pH'].describe()

df_outliers.to_csv('Data Outliers3.csv',index=None)










