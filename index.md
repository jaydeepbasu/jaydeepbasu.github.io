# Portfolio

---

## COVID19 Dashboard - COVID19 Data Analysis
<img src="images/covid19_DataAnalysis.jpg?raw=true"/>

### Project Description
With the world coming to standstill, thousands losing their lives and economies on free fall, COVID19 has hijacked the planet right now.
I was going through COVID19 data one day and thought of creating a dashboard out of my Jupyter notebook, hence started looking out for data from some reliable source and found one where it is updated regularly.
I analyzed data and plotted using Plotly on various parameters like : 
1. Countries with highest number of confirmed/death/recovered new cases.
2. Trend of Active/Recovered/Death cases country wise.
3. Worst hit countries with maximum number of confirmed/death cases.
and many more.. 

FInally deployed the Web App on Heroku to interact and visualize the analysis on COVID 19 spread across the world.

Checkout the project on [Github](https://github.com/jaydeepbasu/covid19-dashboard) 

App is deployed in Heroku, click the link to access it : [Open in Heroku](https://covid19-dashboard--app.herokuapp.com/) 

---


## Multiclass Text Classification - News Dataset
<img src="images/multiclass_Classification_NewsDataset.jpeg?raw=true"/>

### Project Description
Text classification is a supervised learning technique so we’ll need some labeled data to train our model. I am using a public news classification dataset. It’s a manually labeled dataset of news articles which fit into one of 4 classes: Business, SciTech, Sports or World.
The objective is to predict the correct class given a news dataset.
In this notebook, I have used multiple techniques to compare the classification accuracy on the news dataset.
Approach 1 : Using TFIDF and then reducing dimensions using TruncatedSVD.
Approach 2 : Using Word Embeddings like Glove to build my features.
Approach 3 : Using custom trained word embeddings instead of using Glove vectors.

However as the dataset size is small, Glove worked best here.

Checkout the project on [Github](https://github.com/jaydeepbasu/multiClass-text-classification-news-article)

---


## Sentiment Analysis on Tweets
<img src="images/sentiment_analysis_tweet.png?raw=true"/>

### Project Description
Developed a Sentiment Analysis web application and hosted the same on Heroku platform, where we can provide tweets and it tells us the sentimment (whether Positive/Negative).
I have used TFIDF vectorizer and WordNet Lemmatizer by passing POS tags on each word, the model has been trained on number of models like Naive Bayes, SVM and used Logistic Regression algorithms but found Logistic Regression to outperform others, hence selected the same as the final model.
The model has been trained on 1.6 million tweets, the dataset is available on Kaggle.

Checkout the project on [Github](https://github.com/jaydeepbasu/sentiment-analysis-tweet)

App is deployed in Heroku, click the link to access it : [Open in Heroku](https://sentiment-analysis-tweet--app.herokuapp.com/)

---


## Diabetes Prediction App
<img src="images/diabetes_prediction_app.jpg?raw=true"/>

### Project Description
A Web application to predict the onset of diabetes based on diagnostic measures.
Performed cross validation across multiple models and then selected the model with highest accuracy, found that Random Forest Classifier is giving us the highest accuracy, performed hyperparameter tuning on the same further to obtain better result.
The dataset is available on Kaggle.
Also, went ahead and created a docker file which can be used to create a docker image and the same can be used to deploy and host it on any cloud platform whatsoever.

Checkout the project on [Github](https://github.com/jaydeepbasu/diabetes-prediction-app)

App is deployed in Heroku, click the link to access it : [Open in Heroku](https://diabetes-prediction--app.herokuapp.com/)

---


## Fake News Detection
<img src="images/Fake_News_Detection.jpg?raw=true"/>

### Project Description
Developed a model to detect fake news, using NLTK libraries and Decision Tree classifier. I have used the dataset available in Kaggle to train the model.
Also tried out other classification models such as Naive Bayes, Logistic Regression, Random Forest followed up with hyperparameter tuning of each to find out the best performing model.

Checkout the project on [Github](https://github.com/jaydeepbasu/fake-news-detection)

---


## Car Price Prediction
<img src="images/car_price_prediction.jpg?raw=true"/>

### Project Description
Predicting Car Prices and Identifying Important Features impacting car prices using ML algorithms.
Developed two separate models, one to predict car prices, using XGBoost and another model (Linear Regression) to identify the important features that contribute to increase in car price.
I have used the dataset available in Kaggle to train the model.

Also, went ahead and created a docker file which can be used to create a docker image and the same can be used to deploy and host it on any cloud platform whatsoever.

Checkout the project on [Github](https://github.com/jaydeepbasu/car-price-prediction)

App is deployed in Heroku, click the link to access it : [Open in Heroku](https://car-price-prediction--app.herokuapp.com/)

---


## Digit Recognizer
<img src="images/digit_recognizer.png?raw=true"/>

### Project Description
Developed a model to correctly identify digits from a dataset of tens of thousands of handwritten image using SVM. I have used the dataset available in Kaggle to train the model.
Tried with both Linear and Non Linear SVM's but found that Non Linear SVM post hyperparameter tuning is giving better accuracy.

Checkout the project on [Github](https://github.com/jaydeepbasu/digit-recognizer)

---
---