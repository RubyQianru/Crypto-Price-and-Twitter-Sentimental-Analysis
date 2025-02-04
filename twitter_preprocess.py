# Import external libraries
import numpy as np
import pandas as pd
import math
import time
import datetime
# import mplfinance as mpf
from dateutil import parser
import re
# Import utility functions
import util.vader as t

'''
Twitter Sentimental Analysis
'''
def preprocess_twitter_data():
  # Read raw twitter dataset
  tw_data = pd.read_csv('data/twitter_data.csv',index_col=0)

  # If there is missing values, drop these missing values
  tw_data = tw_data.dropna()

  # Extract link values from the **text** column with regex.
  tw_data['text'] = tw_data['text'].apply(
    lambda x: re.sub(r'https?://\S+', '', x).strip()
    )  

  # Remove all "\n" from the **text** column.
  tw_data['text'] = tw_data['text'].replace('\n', '', regex=True)

  # Assert if there is any empty strings for the **text** column
  tw_data = tw_data[tw_data['text'] != '']

  # Drop every row where **text** column is an empty string
  tw_data = tw_data.drop(tw_data[tw_data['text'].isna() | (tw_data['text'].str.strip() == '')].index)

  # Drop rows where column quotes or replies or retweets or bookmarks or favorites is less than 10.
  tw_data = tw_data.drop(
    tw_data[
      (tw_data['quotes'] < 10) | 
      (tw_data['replies'] < 10) | 
      (tw_data['retweets'] < 10) | 
      (tw_data['bookmarks'] < 10) | 
      (tw_data['favorites'] < 10)
    ].index
  )

  # Drop rows where column lang is not "en" (Twitter text is not in English)
  tw_data = tw_data.drop(tw_data[(tw_data['lang'] != 'en')].index)

  # Apply VADER sentiment anaylysis to the twitter dataset.
  tw_data[['vd_negative', 'vd_neutral', 'vd_positive', 'vd_compound']] = tw_data['text'].apply(
    lambda x: pd.Series(t.vader_sentiment(x))
    )

  # Convert current cleaned data to csv
  tw_data.to_csv('./processed/processed_twitter_data.csv', index=False)

  return

def convert2_daily_data():
   # Read processed Twitter dataset
  tw_data = pd.read_csv('./processed/processed_twitter_data.csv')

  # Convert **time** column datatype
  tw_data['time'] = pd.to_datetime(tw_data['time'], format='mixed')
  tw_data = tw_data.sort_values(by='time')

  # Daily mean
  v_day_mean = tw_data.groupby(
    tw_data['time'].dt.floor('d')
    )[['vd_positive', 'vd_negative', 'vd_neutral', 'vd_compound']].mean().reset_index()
  
  # Daily median
  v_day_med = tw_data.groupby(
    tw_data['time'].dt.floor('d')
    )[['vd_positive', 'vd_negative', 'vd_neutral', 'vd_compound']].median().reset_index()
  
  return v_day_mean, v_day_med
