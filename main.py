import sys
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
import util.analysis as a
import util.vader as t

# Import module functions
from price_preprocess import preprocess_price_data, convert2_daily_price
from twitter_preprocess import preprocess_twitter_data, convert2_daily_data
from data_merge import merge_data
from model_train import train_random_forest_model

if __name__ == '__main__': 

  '''
  Price and Twitter sentiment data preprocessing
  '''
  # Bitcoin Price Analysis
  preprocess_price_data()

  # Twitter Sentimental Analysis
  preprocess_twitter_data()

  # Continue Preprocess VADER Twitter Sentiment Data
  v_day_mean, v_day_med = convert2_daily_data()

  # Find Correlation between Bitcoin Price Movemnent VS. Twitter Sentiment
  day_btc = convert2_daily_price()

  # Merge daily btc_data with tw_data using sentiment mean values
  merge_data(day_btc, v_day_mean, v_day_med)

  '''
  Random Forest Model
  '''
  train_random_forest_model()










  

    