
# FinalProject-WorldWineWebML
UTA Data Bootcamp Final Project - Machine Learning (with Wine!)

<p align="center">
  <img width="708" src="images/wine_ml.jpg" alt="WWW Image"><br>
  The World Wine Web
</p>


#### Team Members
Ryan Frescas, Eric Staveley, Eric Tonian

## Project Goal
Our goal is to utilize Natural Language Processing (NLP) for Machine Learning in order to predict a number of dependent variables (price, points, country, varietal) from the description of the wine. In order to do this, we will take the following steps:
  1) ETL: use a combination of existing data from WineEnthusiast and newly scrapped data to create a large enough dataset to train the model.
  2) Textual Analysis: analyze the wine descriptions that will make up our explanatory variable by exploring word counts, performing sentiment analysis, and looking for word associations with word2vec.
  3) Pre-processing and Model selection: prepare data through removal of stop words, tokenization, vectorization, over/under sampling. Then choose variety of models for each of our dependent variables with best theoretical fit, train, test, compare!

## Presentation
Intro - Us, Topic, Goal *Eric T*  
Data - Source, Scrape (ETL) *Ryan*  
Desc. Analysis - Description, Sentiment, word2vec *Ryan*   
ML Models - Price/Points, Variety, Country; how we decided to filter data, what methods did we need to use, why did we pick               this/these models *Eric&Eric*   
Conclusion- what we found (is it useful?), what can we build off of this, what we learned *All*  

## Data Source
1) [Wine Review Dataset](https://www.kaggle.com/zynicide/wine-reviews)- 180k+ wine reviews from WineEnthusiast with variety, location, winery, price, and description
2) Our own additional scrapping of the same source: [WineEnthusiast](https://winemag.com/)

## ETL Process

## Text Analysis

## ML Models
#### Preprocessing

Since we are dealing with a string of text as the explanatory variable, our model pipeline cleans the strings, removes stop words and creates a matrix of token counts using CountVectorizer. That matrix is then transformed to a TF-IDF matrix (TfidfTransformer), which gives different weights to the tokens rather than just a pure count. It is then this TF-IDF matrix that is used in the various regression and classification models below to predict the dependent variables.

#### Regression Analysis - Price

Our first variable to test is price, as that was the original goal of our project. To keep it simple, we first utilize a simple linear regression (just one explanatory variable) at the end of our model pipeline. The results below, while disappointing, are very informative in telling us that this may not be the most useful relationship to explore:
<p align="center"><strong>Simple Linear Regression Model - Price</strong><br>
  Accuracy: 0.08765114373376037<br>
  Mean Squared Error (MSE): 1002.207709822421<br>
  R-squared (R2): 0.4096926120577138
</p>

Just to check for any improvement, we try using the Linear SVR model, which will only allow for points within a specific boundary of tolerance in order to minimize error. The results below, while marginally different, still tell us that we may be barking up the wrong tree:
<p align="center"><strong>LinearSVR Model - Price</strong><br>
  Accuracy: 0.12537641559159662<br>
  Mean Squared Error (MSE): 1464.547758400848<br>
  R-squared (R2): 0.13737107257783931
</p>

Looking at these results of the price models, price does not seem to be accurately predicted by the wine's description. As with many things in life, the cost of something does not always equate to it's qualitative attributes so it makes sense that the description itself does not reveal details of the wine's price.

#### Regression Analysis - Points

Instead of giving up on our regression analysis, we turn to the points variable. This is essentially a 100-point scale rating that the reviewer gives the wine. Using the same exact models as we did for price, we can see a dramatic difference in the model accuracy:
<p align="center"><strong>Simple Linear Regression Model - Points</strong><br>
  Accuracy: 0.7121472485887046<br>
  Mean Squared Error (MSE): 1.9864033844341002<br>
  R-squared (R2): 0.8069739225932919
</p>
<p align="center"><strong>LinearSVR Model - Points</strong><br>
  Accuracy: 0.7401888641421794<br>
  Mean Squared Error (MSE): 2.455881789969409<br>
  R-squared (R2): 0.7613529899278689
</p>
These results tell us that the wine description can potentially give us useful predictions on the rating that the reviewer gives it. Taking a step back this can be reasonably explained, as the general sentiment (i.e. the reviewer's enthusiasm for the wine) within the description ought to relate to what rating the wine receives. 

#### Classification Analysis - Varietal

![nbresults](images/naive_bayes.png)
![tfidflogresults](images/tfidf_logistic_regression.png)
![tfidfscvresults](images/tfidf_svc.png)

#### Classification Analysis - Country

44 countries were representing in the dataset of 171,059 reviews
(insert pic instead) 
The imbalance of the country counts affects the learning portion of the model, and thus the prediction results will bias the countries with more reviews.  Therefore, there is a need to balance the data to improve the accuracy of the models.

imbalanced-learn showed several strategies to address this imbalance:  
 * undersampling: (NearMiss module: to reduce the number of samples in the high freq countries)  
 * oversampling: (SMOTE- Synthetic Minority Oversampling Technique : to smartly generate additional   samples for the under-represented countries, without simple duplication of existing data)

Looking at the data, and shaping the learning dataset to provide best execution times, we limited the data to the top ten countries to model in the three models chosen (to expedite processing and review results)

Modeling Results with Sampling Strategies:  
<p align="center"><strong>CATEGORICAL NAIVE BAYES MODEL</strong><br>
  normal Pipeline Score: 0.8501827040194885<br>
  SMOTE Pipeline Score: 0.8228380024360535<br>
  NearMiss Pipeline Score: 0.40907429963459196
</p>
<p align="center"><strong>TF-IDF LOGISTIC REGRESSION MODEL:</strong><br>
  normal Pipeline Score: 0.8799634591961023<br>
  SMOTE Pipeline Score: 0.8743909866017052<br>
  NearMiss Pipeline Score: 0.48133373934226553
</p>
<p align="center"><strong>TF-IDF LINEAR SVC MODEL:</strong><br>
  normal Pipeline Score: 0.8983861144945189<br>
  SMOTE Pipeline Score: 0.8786236297198539<br>
  NearMiss Pipeline Score: 0.55231425091352
</p>
<p align="center"><strong>OTHER MODELS REVIEWED (scores are the normal Pipeline):</strong><br>
  Categorical Bigram Naive Bayes (Accuracy: 0.7947503799836315)<br>
  Word Count Logistic Regression (Accuracy: 0.8462527768034608)<br>
  Normalized Word Count Logistic Regression (Accuracy: 0.8462527768034608)<br>
  Bigram TF-IDF Logistic Regression (Accuracy: 0.8539985969835145)<br>
  TF-IDF Logistic Regression with Automatic Corpus Specific Stop Words (Accuracy: 0.8578861218285981)<br>
  Linear Discrete Analysis (kernel panic !)<br>
  TF-IDF Linear SVC (Accuracy: 0.881942008651935)
</p>

## Conclusion

