#!/usr/bin/env python
# coding: utf-8

# # Performing Sentiment Analysis on Tweets
# - Following Tutorial: [https://youtu.be/uPKnSq6TaAk]
# - See Also:           [https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=I+like+you.+I+love+you]

# In[3]:


import tweepy
import configparser
import pandas as pd
import snscrape.modules.twitter as sntwitter
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob


# # Initialization of Tokens, Keys, & Twitter API

# In[4]:


config = configparser.ConfigParser()


# In[5]:


config.read('config.ini')


# In[6]:


api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']


# In[7]:


auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth)


# In[8]:


import tweepy
import configparser
import pandas as pd

config = configparser.ConfigParser()

config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth, wait_on_rate_limit=True)


# # Sentiment Analysis

# In[27]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax



# In[112]:


def analyze_sentiment(tweet):
    # Preprocess Tweet
    words = tweet.split(' ')
    words = ['@user' if (w.startswith('@') and len(w) > 1) else w for w in words]
    tweet_preprocessed = " ".join(words)
    #print(f'Tweet (Preprocessed): {tweet_preprocessed}')
    
    # Initialize Model
    roberta_url = 'cardiffnlp/twitter-roberta-base-sentiment'
    model = AutoModelForSequenceClassification.from_pretrained(roberta_url)
    tokenizer = AutoTokenizer.from_pretrained(roberta_url)
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Run Model(Tweet)
    tweet_encoded = tokenizer(tweet_preprocessed, return_tensors='pt')
    #output = model(tweet_encoded['input_ids'], tweet_encoded['attention_mask'])
    output = model(**tweet_encoded)
    
    # Output
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    #for i in range(len(scores)):
    #    print(f'{labels[i]}: {scores[i]}')
        
    return labels, scores


def classify_sentiment(tweets):
    mood_tallies = {
    'Positive': 0,
    'Negative': 0,
    'Neutral': 0
    }
    
    for tweet in tweets:
        labels, scores = analyze_sentiment(tweet)
        curr_scores = zip(labels, scores)
        max_sentiment = max(curr_scores, key=lambda x: x[1])
        mood, mood_rating = max_sentiment[0], max_sentiment[1]
        mood_tallies[mood] += 1
        
    classification = max(mood_tallies, key=mood_tallies.get)
    return classification


def create_user_dataframe(userList):
    
    d = {
        'user': [],
        'mood': [],
        'follower_mood': []
    }
    
    for user in userList:
        f_tweets = get_follower_tweets(user, 5)['tweet'].values
        tweets = twitter_api.get_user(screen_name=user).timeline()
        tweets = [t.text for t in tweets]

        mood = classify_sentiment(f_tweets[:6])
        follower_mood = classify_sentiment(tweets[:6])

        d['user'].append(user)
        d['mood'].append(mood)
        d['follower_mood'].append(follower_mood)
        
    df = pd.DataFrame.from_dict(d, orient='index').T
    return df

def create_user_dataframe2(userList):
    print('running!')
    
    d = {
        'user': [],
        'mood': [],
        'follower_mood': []
    }
    
    for user in userList:
        user_obj = twitter_api.get_user(screen_name=user)
        if user_obj.protected: 
            continue
            
        f_tweets = get_follower_tweets(user, 10)['tweet'].values
        tweets = user_obj.timeline(count=10)
        tweets = [t.text for t in tweets]

        #mood = classify_sentiment(f_tweets[:6])
        #follower_mood = classify_sentiment(tweets[:6])
        mood = np.mean(list(map(lambda x: TextBlob(x).sentiment[0], tweets)))
        follower_mood = np.mean(list(map(lambda x: TextBlob(x).sentiment[0], f_tweets)))

        d['user'].append(user)
        d['mood'].append(mood)
        d['follower_mood'].append(follower_mood)
        
    df = pd.DataFrame.from_dict(d, orient='index').T
    return df




    



def get_follower_tweets(screen_name, max_users):

    # Get User Object

    user_obj = twitter_api.get_user(screen_name=screen_name)
    user_timeline = user_obj.timeline()
    user_followers = twitter_api.get_followers(screen_name=screen_name, count=max_users)

    # Dict for tweets & metadata will be turned into a Pandas dataframe
    
    dict_tweets = {
    'user': [],
    'tweet': [],
    #'source': []
    }

    # For each follower, add all timeline tweets to dict
    
    for curr_user in user_followers:
        if curr_user.protected:
            continue
            
            
        curr_timeline = curr_user.timeline(count=10)
        for tweet in curr_timeline:
            dict_tweets['user'].append(curr_user.screen_name)
            dict_tweets['tweet'].append(tweet.text)
            #dict_tweets['source'].append(tweet.source)


    # Create dataframe
    
    df = pd.DataFrame(dict_tweets)
    return df


def preprocess_tweet(tweet, custom_stopwords, stop_words = stopwords.words("english")):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace('[^\w\s]','')
    preprocessed_tweet = " ".join(word for word in preprocessed_tweet.split() if word not in stop_words)
    preprocessed_tweet = " ".join(word for word in preprocessed_tweet.split() if word not in custom_stopwords)
    preprocessed_tweet = " ".join(Word(word).lemmatize()for word in preprocessed_tweet.split())
    return(preprocessed_tweet)


def compute_twitter_query_mood(screen_name, max_tweets, query_form='from'):
    query = f'({query_form}:{screen_name})'
    tweets = []
    
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) >= max_tweets:
            break
            
        tweets.append(tweet.content)
        
    mood = np.mean(list(map(lambda x: TextBlob(x).sentiment[0], tweets)))
    
    return mood








