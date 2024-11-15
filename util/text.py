import re
from flair.data import Sentence
from flair.nn import Classifier

tagger = Classifier.load('sentiment')

def extract_urls(input:str):
  url_pattern = r'https?://\S+'
  urls = re.findall(url_pattern, text)
  return urls[0] if urls else None

def flair_sentiment(input:str):
  sentence = Sentence(input)
  tagger.predict(sentence)

  if sentence and len(sentence.labels) >= 1 and sentence.labels[0]:
    return sentence.labels[0].value, sentence.labels[0].score 
  else:
    print("String is empty. No result.")
    return

