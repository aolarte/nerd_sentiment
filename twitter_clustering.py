import os
import queue
import threading
import time

import flask
from flask import Flask
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from twython import Twython

APP_KEY = os.environ['APP_KEY']
APP_SECRET = os.environ['APP_SECRET']
ACCESS_TOKEN = os.environ['ACCESS_TOKEN']

MAX_TO_FETCH = 100
MAX_TO_RETURN = 20
MAX_TO_SEED = 1000

SEARCH_TERM = "#WhatBringsMeJoy"

n_features = 1000
idf = True
verbose = False

twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
processed_tweets = []
app = Flask(__name__)
q = queue.Queue()

vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                             min_df=2, stop_words='english',
                             use_idf=idf)
km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1,
            verbose=verbose)


@app.route('/tweets')
def getTweets():
    data = []
    while len(data) < MAX_TO_RETURN and not q.empty():
        data.append(q.get(block=False))
    return flask.jsonify(data)


@app.route('/query')
def getQuery():
    return SEARCH_TERM


def worker():
    while True:
        tweets = twitter.search(q=SEARCH_TERM, count=MAX_TO_FETCH)
        for status in tweets['statuses']:
            tweet_id = status['id']
            tweet_text = status['text']
            if tweet_id not in processed_tweets:
                tweet_features = vectorizer.transform([tweet_text])
                result = km.predict(tweet_features)
                processed_tweets.append(tweet_id)
                cluster = str(result[0])
                q.put({'id': str(tweet_id), 'cluster': cluster})
        time.sleep(2)


if __name__ == '__main__':
    print("Fetching training data")
    training_data = []
    tries = 0
    while len(training_data) < 2000 and tries < 10:
        training_tweets = twitter.search(q=SEARCH_TERM, count=MAX_TO_SEED)
        for training_status in training_tweets['statuses']:
            training_data.append(training_status['text'])
        tries += 1
        time.sleep(2)
    training_features = vectorizer.fit_transform(training_data)
    km.fit(training_features)
    print("K-Means model trained")
    t = threading.Thread(target=worker)
    t.start()
    app.run( host='0.0.0.0', port= 5000)
