import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER analyzer
analyzer = SentimentIntensityAnalyzer()


def vader_sentiment(input:str):
  if input and len(input) > 0:
    vs = analyzer.polarity_scores(input)
    return vs['neg'], vs['neu'], vs['pos'], vs['compound']
  else:
    print("String is empty. No result.")
    return 
