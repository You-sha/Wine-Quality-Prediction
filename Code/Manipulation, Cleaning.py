# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:33:10 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned data.csv')
df.columns

## REMOVING OUTLIERS ##

df_outliers = df[
    ((df.total_sulfur_dioxide < df.total_sulfur_dioxide.quantile(.995))
    &
     (df.total_sulfur_dioxide > df.total_sulfur_dioxide.quantile(.005)))
    &
    ((df.pH < df.pH.quantile(.995))
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
    ((df.volatil_acidity < df.volatil_acidity.quantile(.995))
     &
     df.volatil_acidity > df.volatil_acidity.quantile(.005))]

df_outliers.to_csv('Data Outliers.csv',index=None)

## PLOTS ##

# Bar
my_cmap = plt.get_cmap("magma")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
plt.rcParams["figure.figsize"] = (15,6.75)

plt.subplot(2,2,1)
plt.bar(df_outliers['quality'], df_outliers['alcohol'], color=my_cmap(rescale(df_outliers['quality'])))
plt.ylim(10,15)
plt.xlabel('Quality')
plt.ylabel('Alcohol')

plt.subplots_adjust(wspace=0.2,hspace=0.36)
plt.suptitle('Quality Correlation (Removed Outliers)',fontsize=20)

plt.subplot(2,2,2)
plt.bar(df_outliers['quality'], df_outliers['density'], color=my_cmap(rescale(df_outliers['quality'])))
plt.ylim(0.98,1.045)
plt.xlabel('Quality')
plt.ylabel('Density')

plt.subplot(2,2,3)
plt.bar(df_outliers['quality'], df_outliers['chlorides'], color=my_cmap(rescale(df_outliers['quality'])))
plt.xlabel('Quality')
plt.ylabel('Chlorides')

plt.subplot(2,2,4)
plt.bar(df_outliers['quality'], df_outliers['volatil_acidity'], color=my_cmap(rescale(df_outliers['quality'])))
plt.xlabel('Quality')
plt.ylabel('Volatie Acidity')

plt.savefig('Bar Outliers.png',dpi=600,bbox_inches='tight')

# Histogram
plt.subplot(2,2,1)
n, bins, patches = plt.hist(df_outliers['alcohol'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Alcohol')

plt.subplots_adjust(wspace=0.2,hspace=0.36)
plt.suptitle('Feature Distribution (Removed Outliers)',fontsize=20)

plt.subplot(2,2,2)
n, bins, patches = plt.hist(df_outliers['density'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Density')

plt.subplot(2,2,3)
n, bins, patches = plt.hist(df_outliers['chlorides'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Chlorides')

plt.subplot(2,2,4)
n, bins, patches = plt.hist(df_outliers['volatil_acidity'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Volatile Acidity')

plt.savefig('Distributions (outliers).png',bbox_inches = 'tight',dpi=700)
plt.show()

# Boxplots
plt.subplot(2,2,1)
plt.boxplot(df_outliers['alcohol'])
plt.ylabel('Alcohol')

plt.subplots_adjust(wspace=0.2,hspace=0.36)

plt.subplot(2,2,2)
plt.boxplot(df_outliers['density'])
plt.ylabel('Density')

plt.suptitle('Distributions Boxplots (Removed outliers)',fontsize=20)

plt.subplot(2,2,3)
plt.boxplot(df_outliers['chlorides'])
plt.ylabel('Chlorides')

plt.subplot(2,2,4)
plt.boxplot(df_outliers['volatil_acidity'])
plt.ylabel('Volatile Acidity')

plt.savefig('Distribution of features (Removed Outliers).png',bbox_inches = 'tight',dpi=700)
plt.show()

















