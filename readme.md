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

Let's look at the top four features that impact quality (```alcohol```, ```density```, ```residual_sugar```, and ```total_sulfur_dioxide```):

<img src="https://user-images.githubusercontent.com/123200960/219882985-1b0840ab-54a0-4ccc-ac1e-d85b8aff7a7a.png" width="1000" height="480">

Alcohol and density are difficult to read as their changes are so small. Let's close up by setting a smaller y limit.

<img src="https://user-images.githubusercontent.com/123200960/219883963-9c6cba1a-d0a0-499e-98ac-7d13b952da94.png" width="1000" height="280">

```alcohol``` - Wines with quality between 6 to 8 have the most alcohol. Interestingly, 9s have almost the same alcohol as 3s.

```density``` - Normally distributed. Wines with quality 6 seem to be the most dense. Highest quality wines have the lowest density.

```residual_sugar``` - Similar as density.

```total_sulfur_dioxide``` - Presence seems to be highest in low quality wines, while lowest in high quality ones.

___

Let's take a look at their distributions:

<img src="https://user-images.githubusercontent.com/123200960/219884326-a1bcf08f-c34f-4592-a415-8464a3a81e85.png" width="1000" height="500">

```alcohol``` - Fairly normal distribution. Skewed to the right.

```density```, ```residual_sugar``` - Non normal distribution. There seem to be a lot of outliers. These will probably have to be scaled.

```total_sulfur_dioxide``` - Somewhat normal. Few outliers. Might not need scaling.

___

Since ```alcohol``` is the most correlated feature, it might make sense to have a look at its relation with the other features.

<img src="https://user-images.githubusercontent.com/123200960/219884796-052d80db-a6fd-419e-8cf5-42836474756a.png" width="260" height="600">

Let's also have a look at the relations between it's top 4 highest correlated variables, ```density```, ```residual_sugar```, ```total_sulfur_dioxide``` and ```chlorides``` (I shall not include ```quality``` as it is not continuous).

<img src="https://user-images.githubusercontent.com/123200960/219885039-6c2799c8-69ec-435f-9552-c805348f7103.png" width="1000" height="500">

Wow. This makes me want to try building a model to predict the alcohol amount of a wine as well. I wonder though if the high correlation between density and alcohol could have a negative impact on the quality prediction.

Since all of these features are negatively correlated; we can see that in general the lower the presence of these features, the better the quality tends to be.



*(In progress)*
