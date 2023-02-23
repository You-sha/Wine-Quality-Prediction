# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:13:03 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('winequality-white.csv')
df[['fixed_acidity',"volatil_acidity","citric_acid","residual_sugar",
   "chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density",
   "pH","sulphates","alcohol","quality"]] = df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'].apply(\
                                               lambda x: pd.Series(str(x).split(";")))

df.drop('fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"', axis=1, inplace=True)
sns.heatmap()
df.dtypes
df.corr()
df = df.astype('float')

## DATA EXPLORATION ##

# Heatmaps #

# Quality
plt.figure(figsize=(10, 10))
correlations = pd.DataFrame(df.corr()["quality"].sort_values(ascending=False))
sns.heatmap(correlations,annot=True,square=True,linewidth=2,cbar_kws={'shrink':0.96})
plt.title('Correlation with Quality',fontsize=18, pad=30)
plt.savefig('Heatmap.png',bbox_inches='tight',dpi=600)

# Alcohol
plt.figure(figsize=(10, 10))
correlations = pd.DataFrame(df.corr()["alcohol"].sort_values(ascending=False))
sns.heatmap(correlations,annot=True,square=True,linewidth=2,cbar_kws={'shrink':0.96})
plt.title('Correlation with Alcohol',fontsize=18, pad=30)
plt.savefig('Heatmap_alcohol.png',bbox_inches='tight',dpi=600)


# All
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(),annot=True,square=True,linewidth=2,cbar_kws={'shrink':0.82})
plt.title('Correlation Matrix',fontsize=18, pad=30)
plt.savefig('Heatmap_full.png',bbox_inches = 'tight',dpi=600)

# Relations #

# Alcohol (top 4)
plt.rcParams["figure.figsize"] = (15,6.75)

plt.subplot(2,2,1)
plt.scatter(x=df['alcohol'],y=df['density'],c=df['quality'],cmap='magma')
plt.colorbar(label="Quality", orientation="vertical")
plt.xlabel('Alcohol')
plt.ylabel('Density')

plt.subplots_adjust(wspace=0.1,hspace=0.37)
plt.suptitle('Alcohol Correlation',fontsize=20)

plt.subplot(2,2,2)
plt.scatter(x=df['alcohol'],y=df['residual_sugar'],c=df['quality'],cmap='magma')
plt.colorbar(label="Quality", orientation="vertical")
plt.xlabel('Alcohol')
plt.ylabel('Residual sugar')

plt.subplot(2,2,3)
plt.scatter(df['alcohol'],df['total_sulfur_dioxide'],c=df['quality'],cmap='magma')
plt.colorbar(label="Quality", orientation="vertical")
plt.xlabel('Alcohol')
plt.ylabel('Total Sulfur Dioxide')

plt.subplot(2,2,4)
plt.scatter(df['alcohol'],df['chlorides'],c=df['quality'],cmap='magma')
plt.colorbar(label="Quality", orientation="vertical")
plt.xlabel('Alcohol')
plt.ylabel('Chlorides')

plt.savefig('Correlations with alcohol',bbox_inches = 'tight',dpi=700)
plt.show()

# Quality
my_cmap = plt.get_cmap("magma")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
plt.rcParams["figure.figsize"] = (15,6.75)

plt.subplot(2,2,1)
plt.bar(df['quality'], df['alcohol'], color=my_cmap(rescale(df['quality'])))
plt.ylim(10,15)
plt.xlabel('Quality')
plt.ylabel('Alcohol')

plt.subplots_adjust(wspace=0.2,hspace=0.36)
plt.suptitle('Quality Correlation',fontsize=20)

plt.subplot(2,2,2)
plt.bar(df['quality'], df['density'], color=my_cmap(rescale(df['quality'])))
plt.ylim(0.98,1.045)
plt.xlabel('Quality')
plt.ylabel('Density')

plt.subplot(2,2,3)
plt.bar(df['quality'], df['chlorides'], color=my_cmap(rescale(df['quality'])))
plt.xlabel('Quality')
plt.ylabel('Chlorides')

plt.subplot(2,2,4)
plt.bar(df['quality'], df['volatil_acidity'], color=my_cmap(rescale(df['quality'])))
plt.xlabel('Quality')
plt.ylabel('Volatie Acidity')

plt.savefig('Correlations with quality (bar)',bbox_inches = 'tight',dpi=700)
plt.show()

## Distribution ##
# Boxplot
plt.subplot(2,2,1)
plt.boxplot(df['alcohol'])
plt.ylabel('Alcohol')

plt.subplots_adjust(wspace=0.2,hspace=0.36)

plt.subplot(2,2,2)
plt.boxplot(df['density'])
plt.ylabel('Density')

plt.subplot(2,2,3)
plt.boxplot(df['chlorides'])
plt.ylabel('Chlorides')

plt.subplot(2,2,4)
plt.boxplot(df['volatil_acidity'])
plt.ylabel('Volatile Acidity')

plt.savefig('Correlations with quality (boxplot)',bbox_inches = 'tight',dpi=700)
plt.show()

# Histograms
plt.subplot(2,2,1)
n, bins, patches = plt.hist(df['alcohol'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Alcohol')

plt.subplots_adjust(wspace=0.2,hspace=0.36)
plt.suptitle('Feature Distribution',fontsize=20)

plt.subplot(2,2,2)
n, bins, patches = plt.hist(df['density'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Density')

plt.subplot(2,2,3)
n, bins, patches = plt.hist(df['chlorides'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Chlorides')

plt.subplot(2,2,4)
n, bins, patches = plt.hist(df['volatil_acidity'])
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', my_cmap(c))
plt.xlabel('Volatile Acidity')

plt.savefig('Distributions.png',bbox_inches = 'tight',dpi=700)
plt.show()

df.to_csv('Cleaned data.csv',index=False)



























