# Wine Quality Research


# Project Description
The goal of the project is to analyze the data, find driving features of wine quality, predict quality of wine, and recommend actions to produce good quality of wine.

# Project Goals:

   - Discover drivers of wine quality
   - Use drivers to develop a model to wine quality
   - Offer recommendation to  produce good quality wine

# Initial Questions
* What is the relationship between alcohol and quality?
* Does density play role in determining quality of wine?
* What role does chlorides and density play on quality of wine?
* Is the relationship between volatile_acidity and quality significant?


# The Plan

* Acquire data
    * Acquired data from Data.World Wine Quality Dataset.

* Prepare data
    * Use functions from prepare.py to clean data. 
      * Rename columns names
      * Checked for Nulls, no Null
      * Encode attributes to fit in ML format.
      * split data into train, validate and test (approximatley 56/24/20)

* Explore Data
    * Use graph and hypothesis testing to answer the following initial questions
        * What is the relationship between alcohol and quality?
        * Does density play role in determining quality of wine?
        * What role does chlorides and density play on quality of wine?
        * Is the relationship between volatile_acidity and quality significant?
       
* Develop Model
    * Use driving attributes to create labels
    * Set up baseline prediction
    * Evaluate models on train data and validate data
    * Select the best model based on the highest accuracy 
    * Evaluate the best model on test data to make predictions

* Draw Conclusions

# Data Dictionary
| Feature | Definition |
|:--------|:-----------|
| fixed acidity| most acids involved with wine or fixed or nonvolatile (do not evaporate readily)|
| volatile acidity| the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste|
| citric acid| found in small quantities, citric acid can add 'freshness' and flavor to wines|
| residual sugar| the amount of sugar remaining after fermentation stops|
| chlorides| the amount of salt in the wine|
| free sulfur dioxide| the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion|
| total sulfur dioxide| amount of free and bound forms of S02 Dependents|
| density| the density of water is close to that of water depending on the percent alcohol and sugar content|
| pH| describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic)|
| sulphates| a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant|
| alcohol| the percent alcohol content of the wine|

# Steps to Reproduce
1. Clone this repo 
2. Data can be also be acquired from [Data.World Wine Quality Dataset](https://data.world/food/wine-quality), save a file as 'winequality-red.csv' and 'winequality-white.csv', and put the file into the cloned repo 
3. Run notebook

# Takeaways and Conclusions
    
* Alcohol, chlorides, volatile acidity and density has high correlation with quality
* Alochol has positive correlation with quality, but chlorides, volatile acidity and density have negative correlation with quality
* All models perfomed better than baseline on train and validat data.
* Random forest tree model has the accuracy score of about 58% on test data and beat the baseline accuracy by about 14%.
* We feel comfortable saying Random forest tree is fit for production, until we are able to out perform it with a different model.

# Recommendations
* We recommend to target wines the have an above average alcohol content(around 10.5 +), as well as, keeping `density` and `volatile_acidity` low
 