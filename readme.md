# <p align="center">Predicting Wine Quality</p>
<p align = "center">Using Machine Learning to predict quality of wines, based on the given features in the dataset.</p>
<p align = "center">Tools used: Python (Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn)</p>

#
  
**Table of contents:**

* [TL;DR](#tldr)
* [Exploration](#exploration)
* [Cleaning](#cleaning)
* [Model Building](#model-building)
  - [Model Tuning](#model-tuning)
* [Conclusion](#conclusion)

**Summary:**

We will explore the data, its distributions and correlations, deal with outliers, select features, form hypotheses and test them, try different classification algorithms, and finally tune the model with the best performance.

Result: A Random Forest Classifier model with **71.33% accuracy** in predicting the quality of a wine. After some adjustments with the help of wine professionals, **this model may be able to help wine companies or individuals to accurately judge the quality of their wines.**

<img src="https://user-images.githubusercontent.com/123200960/220903858-cf4573fe-f523-49ef-9bfc-1958c6648fe1.png" width="650" height="500">


___

***Features:***

**```fixed_acidity```** - Fixed acidity of the wine.

**```volatil_acidity```** - Volatile acidity of the wine (I think it is misspelled).

**```citric_acid```** - Amount of citric acid in the wine.

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

We can see that **alcohol** content and **density** have the biggest impact on the quality of a wine. So in general, the higher the alcohol content, the better the quality, and the lower the density the better the quality. Sulphates seem to have the least impact on quality.

Let's look at the top four features that impact quality (```alcohol```, ```density```, ```chlorides```, and ```volatil_acidity```):

![Correlations with quality (bar)](https://user-images.githubusercontent.com/123200960/221917790-192cb048-0536-4d43-9dba-376adda677ac.png)

Since alcohol and density were difficult to read as their changes are so small, I closed up on them by setting a smaller y limit.

```alcohol``` - Wines with quality between 6 to 8 have the most alcohol. Interestingly, 9s have almost the same alcohol as 3s.

```density``` - Normally distributed. Wines with quality 6 seem to be the most dense. Highest quality wines have the lowest density.

```chlorides``` - Since this is negatively correlated, we can see that the best wines have the lowest chlorides.

```volatil_acidity``` - Similar to chlorides.

___

Let's take a look at their distributions: 

![Distributions](https://user-images.githubusercontent.com/123200960/221917934-f160bb9b-6c0a-4b4b-9c00-c6fdaa7c013d.png)

```alcohol``` - Fairly normal distribution. Skewed to the right.

```density```, ```chlorides```,```volatil_acidity``` - Non normal distribution. There seem to be a lot of outliers. These will probably have to be scaled depending on the model.

Let's take a look at their box plots as well:

![Correlations with quality (boxplot)](https://user-images.githubusercontent.com/123200960/220131692-ba355c8b-0158-457c-aeee-d2af1edaa14a.png)

We can see that there are a lot of outliers. These could affect model performance negatively.

___

Since ```alcohol``` is the most correlated feature, it might make sense to have a look at its relation with the other features.

<img src="https://user-images.githubusercontent.com/123200960/219884796-052d80db-a6fd-419e-8cf5-42836474756a.png" width="260" height="600">

Let's also have a look at the relations between it's top 4 highest correlated variables, ```density```, ```residual_sugar```, ```total_sulfur_dioxide``` and ```chlorides``` (I shall not include ```quality``` as it is not continuous).

![Correlations with alcohol](https://user-images.githubusercontent.com/123200960/221918172-918e9d0d-aca8-417a-a397-3e420772ed78.png)

Wow. This makes me want to try building a model to predict the alcohol amount of a wine as well. I wonder though if the high correlation between density and alcohol could have a negative impact on the quality prediction.

Since all of these features are negatively correlated; we can see that in general the lower the presence of these features, the better the quality tends to be.

## <p align="center">Cleaning</p>

I'm going to make models with four different variations of this dataset:

* ***All of the features in the raw data*** - I expect the models built on this to perform the worst. But it should give an obvious baseline performance to beat.

* ***Highly correlated features in the raw data*** - I expect models on this to perform better than the previous one.

* ***All of the features without outliers*** - I would expect this one to perform somewhat similar to the previous one. But I might be surprised.

* ***Highly correlated features without outliers*** - I expect models built on this one to perform the best.

Since I am not exactly the most knowledgable about wines, it is difficult to say at what point outliers can be safely discarded. So I will be trying my best to only remove extreme cases. Still, it should be kept in mind that this model is not going to be the best to use in real-world scenarios.

### Removing Outliers

For most of the highly correlated features, I have discarded values over the .995 percentile, and under the .005 percentile.

The four highest correlated features in the new dataset:

**Bars:**

![Bar Outliers](https://user-images.githubusercontent.com/123200960/220887264-fd434f11-6c6e-49e5-8cb7-f3f41246e909.png)

**Distributions:**

Boxplots:

![Distribution of features (Removed Outliers)](https://user-images.githubusercontent.com/123200960/220887061-defb4740-81a2-45f6-a723-35189c7d2fb1.png)

Histograms:

![Distributions (outliers)](https://user-images.githubusercontent.com/123200960/220887133-1ec398c2-3342-4191-9c33-b135c3ae3b59.png)

The data is now slightly more normalized. Since I will be using Random Forest to build the model, there's no need to scale the data.

## <p align="center">Model Building</p>

First I import the necessary modules, load the original dataset and the one without outliers, and then I select the features that will be used:

<img src="https://user-images.githubusercontent.com/123200960/220891949-2b415146-9e04-41e7-8da6-1c27dc310b23.png" width="700" height="228">

Next, I split them into different training and target sets and import Random Forest Classifier:

<img src="https://user-images.githubusercontent.com/123200960/220893356-030e1c13-3fbb-4878-972a-34062d56d236.png" width="700" height="188">

Training and scoring:

<img src="https://user-images.githubusercontent.com/123200960/220894339-83c9bdff-5e7e-4e68-bdc9-6dcb630fbb86.png" width="700" height="224">

### Observation

Interestingly, the model fitted with the unaltered data with all of the features performs the best (70.61%), while the one with removed outliers and selected highly correlated features performs the worst. So, my hypothesis was completely wrong.

### <p align="center">Model Tuning</p>

Now we are going to take the best model and increase its performance by tuning the hyperparameters. I'll first use RandomizedSearchCV, and then GridSearchCV on the resulting RSCV model, and tune it further.

First I am going to import randomized search, then set a range for the hyperparameters I want to test. And I fit a new random forest into the RSCV and get the best params.

Then, I instantiate a new model using the best RSCV parameters, fit the training data into it and score the test data:

<img src="https://user-images.githubusercontent.com/123200960/220896489-06ecb5c3-5de2-415b-8890-aaeb30c4c492.png" width="700" height="500">

**Result:** The resulting model has a 70.82% score. A 0.21% increase in model performance.

---

Next, I am going to run a grid search for the values around the best RSCV parameters:

<img src="https://user-images.githubusercontent.com/123200960/220899306-d6310fbb-9d1c-47db-b51e-63ea416f8e59.png" width="700" height="350">

**Result:** The final model has a score of 71.33%. An overall increase of 0.72% in performance from the original model.

Confusion Matrix:

![Random Forest CM](https://user-images.githubusercontent.com/123200960/220903858-cf4573fe-f523-49ef-9bfc-1958c6648fe1.png)

## <p align="center">Conclusion</p>

**We have a Random Forest Classifier model with 71.33% accuracy in predicting the quality of a wine.**

---

While I would like to try and find more ways to improve the performance of this model, I have spent entirely too much time and effort on this project. And have now realized that I am extremely dispassionate about wines. I wanted to stop midway, but I couldn't bring myself to just abandon something I started.

I was certainly forced to learn a lot of things because of this project, as this is one of my very first ones. And even though this is not one of my favorite projects, I will remember it gratefully for this.
