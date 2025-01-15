import pandas as pd 
import numpy as np 
import re  
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go  
import time 
import json 
from datetime import datetime,timedelta
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from adv_machine.log import _print , _print_error

def index_to_closest_timeframe_low(input_df, timestamp):
    """
    Optimized function to find rows corresponding to the closest time frame
    that is strictly before the given timestamp, avoiding groupby.

    Parameters
    ----------
    input_df : pandas.DataFrame
        The input DataFrame with a datetime index
    timestamp : str
        The timestamp in the format 'HH:MM:SS'
    
    Returns
    -------
    pandas.DataFrame
        The DataFrame with the rows corresponding to the closest time frame before the timestamp.
    """
    # Extract the date part of the index (without the time)
    dates = input_df.index.normalize()

    # Calculate the timestamp execution time for each day
    time_execution_daily = dates + pd.to_timedelta(timestamp)
 
    valid_mask = input_df.index <= time_execution_daily

    # Filter the DataFrame for valid rows
    filtered_df = input_df[valid_mask]

    # Check if filtered_df has a multi-dimensional index
    if isinstance(filtered_df.index, pd.MultiIndex):
        raise ValueError("filtered_df has a multi-dimensional index")

    # Select the last valid entry for each date
    closest_times = filtered_df.loc[filtered_df.groupby(dates[valid_mask]).apply(lambda x: x.index.max())]

    return closest_times

def compute_returns_from_cumulative_pnl(cumulative_pnl, initial_capital=10000):
    account_value = cumulative_pnl + initial_capital
    returns = 100 * (account_value / account_value.shift(1) - 1)
    returns = returns.dropna().round(8)
    return returns

 
 
def sort_dates(dates):
    dates = sorted(dates,key = lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    return dates   

def dataframe_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a DataFrame into a dictionary with the index as the key and a dict of column-value pairs as values.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be converted.
    
    Returns:
        dict: A dictionary representation of the DataFrame.
    """
    result = {}
    
    # Iterate through the DataFrame using iterrows() to get both index and row data
    for index, row in df.iterrows():
        # Convert row to dictionary and store in result with index as key
        result[index] = row.to_dict()
    
    return result 
def bottom_top_products(scores, top, bottom, reverse=False):
    # Sort the products by score in descending order for top products
    sorted_by_score_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Sort the products by score in ascending order for bottom products
    sorted_by_score_asc = sorted(scores.items(), key=lambda x: x[1])

    # Determine the top and bottom products
    top_products = set([product for product, score in sorted_by_score_desc[:top]])
    bottom_products = set([product for product, score in sorted_by_score_asc[:bottom]])

    # Initialize the result dictionary
    result = {}

    # Assign values based on reverse parameter
    for product in scores.keys():
        if product in top_products:
            result[product] = -1 if reverse else 1
        elif product in bottom_products:
            result[product] = 1 if reverse else -1
        else:
            result[product] = 0

    return result 
def bottom_top_products_resize(scores, top, bottom):
    # Sort the products by score in descending order for top products
    sorted_by_score_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Sort the products by score in ascending order for bottom products
    sorted_by_score_asc = sorted(scores.items(), key=lambda x: x[1])

    # Determine the top and bottom products
    top_products = set([product for product, score in sorted_by_score_desc[:top]])
    bottom_products = set([product for product, score in sorted_by_score_asc[:bottom]])

    # Initialize the result dictionary
    product_to_score = {}

    # Assign 1 for top products, -1 for bottom products, and 0 for the rest
    for product , score in scores.items():
        if product in top_products:
            product_to_score[product] = np.abs(score)
        elif product in bottom_products:
            product_to_score[product] = -np.abs(score)
        else:
            product_to_score[product] = 0
    # Separate positive and negative signals
    pos_signals = {p: s for p, s in product_to_score.items() if s > 0}
    neg_signals = {p: s for p, s in product_to_score.items() if s < 0}
    
    # Normalize positive signals to sum to 0.5
    pos_total = sum(abs(s) for s in pos_signals.values())
    if pos_total > 0:
        pos_signals = {p: (s/pos_total) * 0.5 for p, s in pos_signals.items()}
    
    # Normalize negative signals to sum to -0.5
    neg_total = sum(abs(s) for s in neg_signals.values())
    if neg_total > 0:
        neg_signals = {p: (s/neg_total) * 0.5 for p, s in neg_signals.items()}
    
    # Combine normalized signals
    result = {**pos_signals, **neg_signals}

    return result 
def flatten(dict): 
    keys = list(dict.keys())
    values = list(dict.values())
    if len(keys) > 1 : 
        raise ValueError('We can t flaten the dict that have more than 1 key')
    else : 
        return keys[0],values[0] 




def get_root_directory():
    """
    Returns the root directory of the current Python file.

    Returns:
    str: The root directory containing the current file.
    """
    # Get the absolute path of the current file
    current_dir = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

def extract_integers(input_string):
    """
    This function takes a string as input and returns a list of integers found in the string.
    
    Parameters:
    input_string (str): The string to search for integers.
    
    Returns:
    List[int]: A list of integers found in the input string.
    """
    # Use regex to find all sequences of digits in the input string
    integer_strings = re.findall(r'\d+', input_string)
    
    # Convert the found digit sequences to integers
    integers = [int(num) for num in integer_strings]
    
    return integers  
def format_datetime(input_date):
    # Set the time components to zero to keep only the date part
    return input_date.replace(hour=0, minute=0, second=0, microsecond=0)


def get_moving_average(series, period, method):
    """
    Calculate the moving average of a given series using either the mean or
    exponentially weighted mean.

    Parameters
    ----------
    series : pandas.Series
        The series to calculate the moving average for
    period : int
        The number of periods to include in the moving average
    method : str
        The method to use for calculating the moving average. Either 'MA' for
        the mean or 'EWM' for the exponentially weighted mean

    Returns
    -------
    pandas.Series
        The moving average of the input series
    """
    # Calculate the moving average using the mean
    if method == 'MA':
        return series.rolling(period).mean()
    # Calculate the moving average using the exponentially weighted mean
    elif method == 'EWM':
        return series.ewm(span=period).mean()
    
def is_null(x, epsilon=1e-10):
    if epsilon < 0:
        raise ValueError("Can not use negative epsilon")
    return abs(x) <= epsilon
def z_score(series, period):
    """
    Calculate the z-score of a given series using the rolling mean and
    standard deviation.

    Parameters
    ----------
    series : pandas.Series
        The series to calculate the z-score for
    period : int
        The number of periods to include in the rolling mean and standard
        deviation

    Returns
    -------
    pandas.Series
        The z-score of the input series
    """
    # Calculate the rolling mean
    series_mean = series.rolling(period).mean()
    # Calculate the rolling standard deviation
    series_std = series.rolling(period).std()
    # Calculate the z-score
    series_z_score = (series - series_mean) / series_std
    # Rename the index to the original series name
    return series_z_score.rename(series.name) 
def clamp(x,min,max):
    if x < min : 
        return min 
    elif x > max : 
        return max 
    else : 
        return x 

def adjust_datetime(x):
        if x.second != 0 or x.microsecond != 0:
            return (x + timedelta(minutes=1)).replace(second=0, microsecond=0)
        else:
            return x 


def get_name_config(config, exclude_keys=None):
    if not config:
        return ''
    if exclude_keys is None:
        exclude_keys = ['product']  # Exclude 'product' by default

    # Extract the main key from the config dictionary
    main_key = list(config.keys())[0]

    # Initialize parts list with main key
    parts = [main_key]

    # Extract the sub-dictionary
    config_main = config[main_key]

    # This function extracts parts in the specified order
    def extract_parts(config_section):
        parts = []

        # Process 'transformation' section if it exists
        if 'transformation' in config_section:
            for trans_key, trans_value in config_section['transformation'].items():
                parts.append(trans_key)
                for key, value in trans_value.items():
                    if key in exclude_keys:
                        continue
                    if isinstance(value, str):
                        parts.append(f"{key}_{value}")
                    else:
                        parts.append(f"{key}{value}")

        # Process 'feature' section if it exists
        if 'feature' in config_section:
            for key, value in config_section['feature'].items():
                if key in exclude_keys:
                    continue
                if isinstance(value, str):
                    parts.append(f"{key}_{value}")
                else:
                    parts.append(f"{key}{value}")

        # Process 'collect' section if it exists
        if 'collect' in config_section:
            collect_section = config_section['collect']
            # Add 'type' value directly
            if 'type' in collect_section and 'type' not in exclude_keys:
                parts.append(collect_section['type'])
        return parts

    sub_parts = extract_parts(config_main)
    parts.extend(sub_parts)

    # Join parts with underscores
    name = "_".join(parts)
    return name



def get_target_name(df):
    for feature in df.columns : 
        if 'target' in feature : 
            return feature 
    raise ValueError(f'utils.get_target_name() . target name not found in dataframe. Columns of the dataframe {df.columns}') 


def add_range_position(series,period):
        range_position = series.copy(deep=True).rolling(period).apply(
            lambda x: np.interp(x.iloc[-1], [x.min(), x.max()], [-1, 1])
        )
        return range_position 



def sort_date_index(input):
    """
    Sort a given DataFrame by its index in ascending order.

    Parameters
    ----------
    input : pandas.DataFrame
        The DataFrame to sort

    Returns
    -------
    pandas.DataFrame
        The sorted DataFrame
    """
    # Sort the DataFrame by its index
    df = input.sort_index()
    # Return the sorted DataFrame
    return df
