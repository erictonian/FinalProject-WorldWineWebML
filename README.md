# FinalProject-WorldWineWebML
UTA Data Bootcamp Final Project - Machine Learning (with Wine!)

#### Team Members
Ryan Frescas, Eric Staveley, Eric Tonian

## Project Goal
Our goal is to utilize Natural Language Processing (NLP) for Machine Learning in order to predict rating and/or price from the description of the wine. In order to do this we will take the following steps:
  1) Gather data -- use a combination of existing data from WineEnthusiast and newly scrapped data to create a large enough        dataset to train the model.
  2) Pre-process data -- clean wine description through removal of punctuation, tokenization, removal of stop words,                stemming/lemmatizing, vectorizing with TF-IDF.
  3) Model selection -- choose models with best theoretical fit (such as Multinomial Naive Bayes), train, test, compare!

## Data Source
1) Wine Review Dataset- 180k+ wine reviews with variety, location, winery, price, and description https://www.kaggle.com/zynicide/wine-reviews
2) Our own scrapping of the same source to add new reviews to the existing dataset (from https://winemag.com/)
