import numpy as np
import pandas as pd


def analyze_returns(daily_data):
  """Analyze price returns and detect outliers."""
  # Calculate returns
  daily_data['rtn'] = daily_data['Close'].pct_change()
  
  # Calculate rolling statistics
  df_rolling = daily_data[['rtn']].rolling(window=5).agg(["mean", "std"])
  df_rolling.columns = ["mean", "std"]
  df_rolling = df_rolling.dropna()
  
  # Calculate bounds
  N_SIGMAS = 2
  df_rolling["upper"] = df_rolling["mean"] + N_SIGMAS * df_rolling["std"]
  df_rolling["lower"] = df_rolling["mean"] - N_SIGMAS * df_rolling["std"]
  
  # Identify outliers
  analysis_df = daily_data[['rtn']].join(df_rolling, how="left")
  analysis_df["outlier"] = (
      (analysis_df["rtn"] > analysis_df["upper"]) | 
      (analysis_df["rtn"] < analysis_df["lower"])
  )
  
  return analysis_df


def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset."""
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Moving averages
    df['SMA_5'] = df['price'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['price'].rolling(window=10, min_periods=1).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['price'].rolling(window=10, min_periods=1).mean()
    bb_std = df['price'].rolling(window=10, min_periods=1).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)  # Fill initial NaN values
    
    # MACD
    exp1 = df['price'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['price'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Volatility
    df['volatility'] = df['price'].pct_change().rolling(window=10, min_periods=1).std() * 100
    
    return df


def detect_patterns(df):
    """Detect candlestick patterns."""
    df = df.copy()
    
    # Ensure we have required columns
    required_columns = ['open', 'close', 'dayHigh', 'dayLow']
    if not all(col in df.columns for col in required_columns):
        print("Warning: Missing required columns for pattern detection. Skipping pattern detection.")
        return df
    
    # Doji pattern
    body = abs(df['close'] - df['open'])
    wick = df['dayHigh'] - df['dayLow']
    df['Doji'] = body <= (wick * 0.1)
    
    # Hammer pattern
    upper_wick = df['dayHigh'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['dayLow']
    df['Hammer'] = (lower_wick > (1.5 * body)) & (upper_wick < body)
    
    return df


def calculate_correlation(x, y):
    """Calculate correlation coefficient value."""
    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient