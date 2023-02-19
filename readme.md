# <p align="center">Predicting Wine Quality</p>
<p align = "center">Using basic ML models to predict quality of wines, based on the given features in the dataset.</p>

<p align = "center">(In progress)</p>

___

***Features:***

**```fixed_acidity```** - Fixed acidity of the wine.

**```volatil_acidity```** - Volatile acidity of the wine (I think it is misspelled).

**```citric_acid```** - Amount of cirtic acid in the wine.

**```residual_sugar```** - Amount of residual sugar in the wine.

**```chlorides```** - Amount of Chlorides in the wine.

**```free_sulfur_dioxide```** - The amount of Sulfur Dioxide in the wine that is free.

**```total_sulfur_dioxide```** - Amount of Sulfur Dioxide present in the wine.

**```density```** - Density of the wine

**```pH```** - pH level of the wine.

**```sulphates```** - Amount of sulphates in the wine.

**```alcohol```** - Alcohol content in the wine.

#

**```quality```** - Quality of the wine. This is our target variable.

#

## <p align="center">Exploration</p>

Opening the dataframe using the Variable Explorer in Spyder:

<img src="https://user-images.githubusercontent.com/123200960/219403282-85543b7a-e141-41b8-ae06-c784c575492a.png" width="600" height="500">


All of the features are dumped into a single column, seperated by semicolons. First I seperate it into different columns by applying a lambda function, then I drop this initial column.

<img src="https://user-images.githubusercontent.com/123200960/219408628-c3240412-dc77-4842-8453-8f27ae1f2906.png" width="600" height="500">

Much better (and now actually usable).

---

First let's look at the correlation between the features.

<img src="https://user-images.githubusercontent.com/123200960/219882675-0b1d2a57-c6ad-42f9-a748-ee97a4b05727.png" width="550" height="500">

Looking at the correlation between just the target variable and features:

<img src="https://user-images.githubusercontent.com/123200960/219882758-0c0256dc-0e4f-4b59-a6d7-4a0fa9e997d6.png" width="260" height="600">

We can see that **alcohol** content and **density** have the biggest impact on the quality of a wine. The higher the alcohol content, the better the quality, and the lower the density the better the quality. Sulphates seem to have the least impact on quality.

Let's look at the top four features that impact quality (```alcohol```, ```density```, ```chlorides```, and ```volatil_acidity```):

<img src="https://user-images.githubusercontent.com/123200960/219965143-63abde58-8d42-49af-8054-2d5e95ef7812.png" width="1000" height="480">

Since alcohol and density were difficult to read as their changes are so small, I closed up on them by setting a smaller y limit.

```alcohol``` - Wines with quality between 6 to 8 have the most alcohol. Interestingly, 9s have almost the same alcohol as 3s.

```density``` - Normally distributed. Wines with quality 6 seem to be the most dense. Highest quality wines have the lowest density.

```chlorides``` - Since this is negatively correlated, we can see that the best wines have the lowest chlorides.

```volatil_acidity``` - Similar to chlorides.

___

Let's take a look at their distributions:

<img src="https://user-images.githubusercontent.com/123200960/219965353-2a5fe6a5-017d-4aea-8254-d3c7abccb9ac.png" width="1000" height="500">

```alcohol``` - Fairly normal distribution. Skewed to the right.

```density```, ```chlorides```,```volatil_acidity``` - Non normal distribution. There seem to be a lot of outliers. These will probably have to be scaled depending on the model.

Let's also have a look at their box plots:

![Correlations with quality (boxplot)](https://user-images.githubusercontent.com/123200960/219965482-048eaf73-5ca1-442c-9900-ed552002636d.png)

We can see that there are a lot of outliers. These might affect model performance negatively.

___

Since ```alcohol``` is the most correlated feature, it might make sense to have a look at its relation with the other features.

<img src="https://user-images.githubusercontent.com/123200960/219884796-052d80db-a6fd-419e-8cf5-42836474756a.png" width="260" height="600">

Let's also have a look at the relations between it's top 4 highest correlated variables, ```density```, ```residual_sugar```, ```total_sulfur_dioxide``` and ```chlorides``` (I shall not include ```quality``` as it is not continuous).

<img src="https://user-images.githubusercontent.com/123200960/219885039-6c2799c8-69ec-435f-9552-c805348f7103.png" width="1000" height="500">

Wow. This makes me want to try building a model to predict the alcohol amount of a wine as well. I wonder though if the high correlation between density and alcohol could have a negative impact on the quality prediction.

Since all of these features are negatively correlated; we can see that in general the lower the presence of these features, the better the quality tends to be.



*(In progress)*
