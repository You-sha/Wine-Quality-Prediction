# Predicting Wine Quality
Using basic ML models to predict quality of wines, based on the given features in the dataset.

### Features: 

**fixed_acidity** - Fixed acidity of the wine.

**volatil_acidity** - Volatile acidity of the wine (I think it is misspelled).

**citric_acid** - Amount of cirtic acid in the wine.

**residual_sugar** - Amount of residual sugar in the wine.

**chlorides** - Amount of Chlorides in the wine.

**free_sulfur_dioxide** - The amount of Sulfur Dioxide in the wine that is free.

**total_sulfur_dioxide** - Amount of Sulfur Dioxide present in the wine.

**density** - Density of the wine

**pH** - pH level of the wine.

**sulphates** - Amount of sulphates in the wine.

**alcohol** - Alcohol content in the wine.

**quality** - Quality of the wine. This is our target variable.

## Exploration of data
Looking at the correlation between the target variable and the given features.

<img src="https://user-images.githubusercontent.com/123200960/219378258-ca0418ce-094b-4be3-b933-bcd18289bc2d.png" width="300" height="600">

We can see that alcohol content and density have the biggest impact on the quality of a wine. The higher the alcohol content, the better the quality, and the lower the density the better the quality. Residual sugars seem to have the least impact on quality.

## Evaluation of KNN = 2
**(Values rounded to the 2nd decimal)**

MSE = 0.31

MAE = 0.58

Variance in y = 0.78

Accuracy = 0.82 

<img src="https://user-images.githubusercontent.com/123200960/218274550-a6f6bf4f-5d0c-4f7e-b03f-1922678b2a5f.png" width="600" height="400">

### The accuracy of other models I tried before this one:

Linear Regression:  0.29

Logistic Regression (liblinear): 0.51

(KNN, n=5): 0.61
