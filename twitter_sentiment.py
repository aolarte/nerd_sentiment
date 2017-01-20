import os
import queue
import threading
import time

import flask
from flask import Flask
from nltk.sentiment import vader
from twython import Twython

APP_KEY = os.environ['APP_KEY']
APP_SECRET = os.environ['APP_SECRET']
ACCESS_TOKEN = os.environ['ACCESS_TOKEN']

MAX_TO_FETCH = 100
MAX_TO_RETURN = 20

SEARCH_TERM = "#NerdSentiment"

sia = vader.SentimentIntensityAnalyzer()
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
results = {}
app = Flask(__name__)
q = queue.Queue()


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
            if tweet_id not in results:
                scores = sia.polarity_scores(tweet_text)
                results[tweet_id] = scores
                q.put({'id': str(tweet_id), 'scores': scores})
        time.sleep(2)


if __name__ == '__main__':
    t = threading.Thread(target=worker)
    t.start()
    app.run( host='0.0.0.0', port= 5001)
