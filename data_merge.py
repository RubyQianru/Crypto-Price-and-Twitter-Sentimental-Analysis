# Import external libraries
import numpy as np
import pandas as pd
import math

def merge_data(day_btc, v_day_mean, v_day_med):
  # Merge daily btc_data with tw_data using sentiment mean values
  day_btc_tw = pd.merge(day_btc, v_day_mean, on='time')

  day_btc_tw = day_btc_tw.rename(columns={
    "vd_positive": "vd_positive_mean",
    'vd_neutral': "vd_neutral_mean",
    "vd_negative": "vd_negative_mean",
    "vd_compound": "vd_compound_mean",
  })

  # Merge btc_data with tw_data using sentiment median values
  day_btc_tw = pd.merge(day_btc_tw, v_day_med, on='time')

  # Rename columns
  day_btc_tw = day_btc_tw.rename(columns={
    "vd_positive": "vd_positive_med",
    'vd_neutral': "vd_neutral_med",
    "vd_negative": "vd_negative_med",
    "vd_compound": "vd_compound_med",
  })

  # Save the daily data to a CSV file if needed
  day_btc_tw = day_btc_tw.dropna()
  day_btc_tw.to_csv('./processed/day_btc_tw.csv', index=False)
