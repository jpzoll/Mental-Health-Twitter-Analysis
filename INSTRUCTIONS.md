# CIS563_FinalProject

# Joseph Zoll and Prarthana Poojara

# Final Project for Intro To Data Science at Syracuse University 🍊

# Instructions For Reading In Order:

### General Files

- Cookbook.py - This holds all of the main functions created with the use of the Twitter API and NLTK. The 2 most important functions in the Cookbook are:
  1. create_user_dataframe2(..)
  2. get_follower_tweets(..)
- Config.ini - Stores the API keys needed to access & authenticate the Twitter API on behalf our developer account
- TWITTER_API_TEMPLATE.ipynb - Template for authenticating Twitter. Uses Config.ini and tweepy

### Needed Libraries

- tweepy
- configparser
- Pandas
- NLTK
  - nltk.corpus
- matplotlib (import matplotlib.pyplot as plt)
- Textblob (from textblob import TextBlob)
- NumPy
- sklearn (from sklearn.cluster import KMeans)

# For Section 1 | Comparative Analysis:

### Main Files (IN ORDER)

1. gathering_mental_health_tweets.ipynb
2. preprocessing_tweets.ipynb
3. calculating_sentiment.ipynb

### Data

1. data/mental_health_tweets.csv
2. data/processed_data.csv

# For Section 2 | K-Means Clustering

### Main Files (IN ORDER)

1. preprocessing_clustering.ipynb
2. k_means_clustering.ipynb

### Data

1. data/user_accounts.csv
2. data/mood_and_f_moods2.csv
