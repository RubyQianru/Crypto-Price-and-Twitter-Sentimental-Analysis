import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import pandas as pd


def positive_negative_distribution_bar(time_data, negative_data, positive_data, width, title):
  """Create bar charts to visualize distribution between positive vs. negative Flair Twitter sentiment data."""
  plt.figure(figsize=(15, 6))

  plt.bar(time_data, negative_data, width, alpha=0.7, label='Negative')
  plt.bar(time_data, positive_data, width, bottom=negative_data, color='purple', alpha=0.7, label='Positive')

  plt.title(title, fontsize=20)
  plt.xlabel('Time')
  plt.ylabel('Count')
  plt.legend()

  plt.gcf().autofmt_xdate()
  plt.tight_layout()
  plt.show()


def sentiment_price_scatter(
  price, sentiment_1, sentiment_2,
  title_1:str, x_label_1:str, y_label_1:str,
  title_2:str, x_label_2:str, y_label_2:str,
  ):
  """Create scatter charts to visualize correlation between prices vs. sentiment data."""

  f, ax = plt.subplots(1, 2, figsize=(15, 6))

  ax[0].scatter(price, sentiment_1)
  ax[0].set_xlabel(x_label_1)
  ax[0].set_ylabel(y_label_1)
  ax[0].set_title(title_1)

  ax[1].scatter(price, sentiment_2)
  ax[1].set_xlabel(x_label_2)
  ax[1].set_ylabel(y_label_2)
  ax[1].set_title(title_2)

  plt.gcf().autofmt_xdate()
  plt.tight_layout()
  plt.show()


def sentiment_price_delta_scatter(
  price, price_delta, sentiment, sentiment_delta,
  title_1:str, x_label_1:str, y_label_1:str,
  title_2:str, x_label_2:str, y_label_2:str,
  ):
  """Create scatter charts to visualize correlation between prices vs. sentiment data."""

  f, ax = plt.subplots(1, 2, figsize=(15, 6))

  ax[0].scatter(price, sentiment)
  ax[0].set_xlabel(x_label_1)
  ax[0].set_ylabel(y_label_1)
  ax[0].set_title(title_1)

  ax[1].scatter(price_delta, sentiment_delta)
  ax[1].set_xlabel(x_label_2)
  ax[1].set_ylabel(y_label_2)
  ax[1].set_title(title_2)

  plt.gcf().autofmt_xdate()
  plt.tight_layout()
  plt.show()


def plot_candlestick(daily_data):
  """Create candlestick chart with indicators."""
  # Define style
  mc = mpf.make_marketcolors(
      up='green', down='red', edge='inherit',
      volume='in', wick={'up': 'green', 'down': 'red'}
  )
  style = mpf.make_mpf_style(
      marketcolors=mc, gridstyle='--', y_on_right=False
  )
  
  # Define additional plots
  apds = [
      mpf.make_addplot(daily_data['Close'], color='purple', width=1.0, 
                      linestyle='--', ylabel='Actual Price'),
      mpf.make_addplot(daily_data['SMA5'], color='blue', width=0.8, 
                      ylabel='SMA5'),
      mpf.make_addplot(daily_data['SMA10'], color='red', width=0.8, 
                      ylabel='SMA10')
  ]
  
  # Create plot
  fig, axes = mpf.plot(
      daily_data,
      type='candle',
      style=style,
      volume=True,
      addplot=apds,
      title='Bitcoin Price every 6 hours',
      panel_ratios=(3, 1),
      figsize=(15, 10),
      returnfig=True
  )
  
  # Add legend
  handles = [
      plt.Line2D([], [], color='purple', linestyle='--', label='Actual Price'),
      plt.Line2D([], [], color='blue', label='SMA5'),
      plt.Line2D([], [], color='red', label='SMA10')
  ]
  axes[0].legend(handles=handles, loc='best', fontsize='small')
  
  return fig, axes


def plot_returns(analysis_df):
  """Plot returns analysis with outliers."""
  fig, ax = plt.subplots(figsize=(15, 6))  # Matched size with technical analysis plots
  
  # Plot returns and bounds
  analysis_df[["rtn", "upper", "lower"]].plot(ax=ax, alpha=0.7)
  
  # Highlight outliers
  ax.scatter(
      analysis_df.loc[analysis_df["outlier"]].index,
      analysis_df.loc[analysis_df["outlier"], "rtn"],
      color="black", label="Outlier", zorder=5
  )
  
  # Customize plot
  ax.set_title("Bitcoin Returns with Outliers", fontsize=14)
  ax.set_xlabel("Date")
  ax.set_ylabel("Returns")
  ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
  
  sns.despine()
  plt.tight_layout()
  
  return fig, ax


def plot_technical_analysis(df, include_patterns=False):
  """Create comprehensive technical analysis plot."""
  # Ensure we have a datetime index
  if not isinstance(df.index, pd.DatetimeIndex):
      if 'time' in df.columns:
          df = df.set_index('time')
      else:
          df.index = pd.to_datetime(df.index)
  
  fig = plt.figure(figsize=(15, 20))
  
  # Price and indicators plot
  ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
  ax1.plot(df.index, df['price'], label='Price', color='blue', alpha=0.7)
  ax1.plot(df.index, df['SMA_5'], label='5-period SMA', color='orange')
  ax1.plot(df.index, df['SMA_10'], label='10-period SMA', color='red')
  ax1.plot(df.index, df['BB_upper'], label='BB Upper', color='gray', linestyle='--')
  ax1.plot(df.index, df['BB_lower'], label='BB Lower', color='gray', linestyle='--')
  ax1.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.1, color='gray')
  
  # Plot patterns if requested and available
  if include_patterns and 'Doji' in df.columns and 'Hammer' in df.columns:
      doji_mask = df['Doji'] == True
      hammer_mask = df['Hammer'] == True
      
      if doji_mask.any():
          ax1.scatter(df.index[doji_mask], df.loc[doji_mask, 'price'], 
                      marker='^', color='green', label='Doji', s=100)
      if hammer_mask.any():
          ax1.scatter(df.index[hammer_mask], df.loc[hammer_mask, 'price'], 
                      marker='v', color='red', label='Hammer', s=100)
  
  ax1.set_title('Bitcoin Price with Technical Indicators')
  ax1.set_ylabel('Price (USD)')
  ax1.legend()
  ax1.grid(True)
  
  # RSI plot
  ax3 = plt.subplot2grid((5, 1), (2, 0))
  ax3.plot(df.index, df['RSI'], color='purple')
  ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
  ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
  ax3.fill_between(df.index, 70, df['RSI'], 
                    where=(df['RSI'] >= 70), color='r', alpha=0.3)
  ax3.fill_between(df.index, 30, df['RSI'], 
                    where=(df['RSI'] <= 30), color='g', alpha=0.3)
  ax3.set_ylabel('RSI')
  ax3.grid(True)
  
  # MACD plot
  ax4 = plt.subplot2grid((5, 1), (3, 0))
  ax4.plot(df.index, df['MACD'], label='MACD', color='blue')
  ax4.plot(df.index, df['Signal_Line'], label='Signal Line', color='red')
  ax4.bar(df.index, df['MACD_Histogram'], color='gray', alpha=0.3)
  ax4.set_ylabel('MACD')
  ax4.legend()
  ax4.grid(True)
  
  # Volatility plot
  ax5 = plt.subplot2grid((5, 1), (4, 0))
  ax5.plot(df.index, df['volatility'], label='Volatility', color='magenta')
  ax5.set_ylabel('Volatility (%)')
  ax5.legend()
  ax5.grid(True)
  
  plt.tight_layout()
  return fig

def plot_correlations(correlations, title:str = "Correlation of Features with Price "):
  """Create correlations bar chart"""
  plt.figure(figsize=(15, 25))

  plt.subplot(5, 1, 5)
  price_correlations = correlations.sort_values(ascending=False)
  price_correlations.plot(kind='bar', color='teal', width=0.8)
  plt.title(title, fontsize=16)
  plt.xlabel('Features', fontsize=14)
  plt.ylabel('Correlation Coefficient', fontsize=14)
  plt.tight_layout()

  plt.show()