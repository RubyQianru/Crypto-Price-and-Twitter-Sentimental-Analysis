import re
from flair.data import Sentence
from flair.nn import Classifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Flair tagger
tagger = Classifier.load('sentiment')
# VADER analyzer
analyzer = SentimentIntensityAnalyzer()


def flair_sentiment(input:str):
  if (len(input) > 0):
    sentence = Sentence(input)
    tagger.predict(sentence)
    return sentence.labels[0].value, sentence.labels[0].score 
  else:
    print("String is empty. No result.")
    return


def vader_sentiment(input:str):
  if input and len(input) > 0:
    vs = analyzer.polarity_scores(input)
    return vs['neg'], vs['neu'], vs['pos'], vs['compound']
  else:
    print("String is empty. No result.")
    return 
