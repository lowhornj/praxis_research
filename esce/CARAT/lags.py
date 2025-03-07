import statsmodels.api as sm
import pandas as pd

def optimal_lag(series,lags=5):
    """
    Finds the optimal lag for a time series based on autocorrelation.

    Args:
        series (pd.Series): The time series data.

    Returns:
        int: The optimal lag.
    """
    autocorrelation_values = sm.tsa.stattools.acf(series)
    significant_lags = np.where(np.abs(autocorrelation_values) > 1.96/np.sqrt(len(series)))[0]

    if len(significant_lags) > 0:
      return significant_lags[1] # Return the first significant lag (excluding lag 0)
    else:
      return 0

def find_optimal_lags_for_dataframe(df):
    """
    Finds the optimal lag for each column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are column names and values are optimal lags.
    """
    optimal_lags = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            optimal_lags.append( optimal_lag(df[col]).item())
    return optimal_lags

from collections import Counter

def most_frequent(list_):
  """
    Finds the most frequent element in a list.

    Args:
      list_: The input list.

    Returns:
      The most frequent element in the list.
  """
  if not list_:
    return None
  count = Counter(list_)
  return count.most_common(1)[0][0]
n_lags = most_frequent(find_optimal_lags_for_dataframe(df))+1
