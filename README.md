# IMDB Dataset of 50K Movie Reviews

**Author:** SamiUllah568

## About Dataset

The IMDB dataset contains 50,000 movie reviews for natural language processing or text analytics. It is widely used for sentiment analysis, particularly in the field of Natural Language Processing (NLP). The dataset is labeled as positive or negative, making it suitable for binary classification tasks.

## Dataset
The dataset used in this project can be accessed from Kaggle:  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


## Context

The IMDB dataset offers a substantial amount of data, allowing for more accurate and robust sentiment classification models compared to smaller datasets used in earlier benchmarks. The primary objective is to classify each review as either positive or negative.

## Aim

The primary aim of this project is to predict the sentiment of movie reviews using classification algorithms. Specifically, the goals are to:

1. Preprocess the text data (tokenization, removing stop words, stemming, etc.).
2. Train a model to classify movie reviews as positive or negative.
3. Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
4. Experiment with different algorithms like Logistic Regression, Random Forest, etc.

## Libraries Used

```python
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings('ignore')
```

## Data Loading

```python
movies = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
movies = movies.sample(10000)
```

## Data Preprocessing

- Handling HTML tags
- Converting to lower case
- Removing stop words
- Removing punctuation
- Tokenization
- Stemming

## Model Training

Several models were trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest
4. XGBoost
5. Support Vector Classifier

## Model Evaluation

The models were evaluated using accuracy, precision, recall, and F1-score. The Logistic Regression model showed the best performance with an accuracy of 85.6% on the test set.

## Model Saving and Loading

The trained models and vectorizers were saved using `pickle` for future use.

## Prediction Function

A function was created to predict the sentiment of new movie reviews using the trained Logistic Regression model.

```python
def prediction(text):
    text = text.lower()
    text = remove_pun(text)
    text = tokenize_text(text)
    text = remove_stopwords(text)
    text = stem_word(text)
    text = " ".join(text)
    vectors = bow.transform([text])
    prediction = log_reg.predict(vectors)
    return encode.inverse_transform(prediction)[0]
```

Example predictions:

```python
text = "This is very bad movie and I really dislike this movie"
print(prediction(text))  # Output: 'negative'

text = "This is very good movie and I really like and love this movie"
print(prediction(text))  # Output: 'positive'
```