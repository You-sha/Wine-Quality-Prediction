# Predicting Wine Quality
Using basic ML models to predict quality of wines, based on the given features in the dataset.

***Features:***

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

## Cleaning data

Opening the dataframe using the Variable Explorer in Spyder:
<img src="https://user-images.githubusercontent.com/123200960/219403282-85543b7a-e141-41b8-ae06-c784c575492a.png" width="600" height="500">


All of the features are dumped into a single column, seperated by semicolons. First I seperate it into different columns by applying a lambda function, then I drop this initial column.

<img src="https://user-images.githubusercontent.com/123200960/219408628-c3240412-dc77-4842-8453-8f27ae1f2906.png" width="600" height="500">

Much better (and now actually usable).

## Exploration of data
Looking at the correlation between the target variable and the given features.

<img src="https://user-images.githubusercontent.com/123200960/219378258-ca0418ce-094b-4be3-b933-bcd18289bc2d.png" width="300" height="600">

We can see that alcohol content and density have the biggest impact on the quality of a wine. The higher the alcohol content, the better the quality, and the lower the density the better the quality. Residual sugars seem to have the least impact on quality.

## Models
I split the data into features and target, and then into a training and a test set. Then I fit it into the models.
### Logistic Regression
<img src="https://user-images.githubusercontent.com/123200960/219413997-99d93bc2-31f2-4540-a809-03f87ce404f8.png" width="600" height="400">




### K-Neighbors Classifier
<img src="https://user-images.githubusercontent.com/123200960/218274550-a6f6bf4f-5d0c-4f7e-b03f-1922678b2a5f.png" width="600" height="400">

**(Values rounded to the 2nd decimal)**

MSE = 0.31

MAE = 0.58

Variance in y = 0.78

Accuracy = 0.82 
