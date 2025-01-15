import pandas as pd 
import numpy as np 

import adv_machine.utils as ut 
import adv_machine.get_prices as gp 
from adv_machine.ohlc_to_feature import ohlc_to_feature  
 
from adv_machine.log import _print, _print_error 



def feature_computer(configs, return_ohlc=False, verbose=0):
    """
    Compute features given a list of configurations.

    Parameters
    ----------
    configs : list of dict
        The configurations for the features.
    config_freq_calculation : dict
        The frequency configuration for the features.
    return_ohlc : bool, optional
        Whether to return the OHLC data along with features. Defaults to False.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    df : pd.DataFrame or tuple
        The dataframe with computed features, or tuple (df, labels_to_ohlc) if return_ohlc=True
    """
    _print('----------------- Feature Computer ft.feature_computer -------------', 2, verbose)

    labels_to_ohlc = {}
    concat = []

    for config in configs:
        feature, config_feature = ut.flatten(config)
        _print(f'Computing feature {feature}', 1, verbose)
        try:
            # Get OHLC data
            config_feature_calculation = config_feature['feature']
            config_collect = config_feature['collect']
            
            product = config_collect['product']




            # Get or fetch OHLC data
            if product in labels_to_ohlc:
                _print(f'Found cached OHLC data for {product}', 2, verbose)
                ohlc = labels_to_ohlc[product].copy(deep=True)
                _print(f'Cached OHLC data shape: {ohlc.shape if hasattr(ohlc, "shape") else len(ohlc)}', 2, verbose)
            else:
                _print(f'No cached data found for {product}, fetching new data...', 2, verbose)
                try:
                    ohlc = gp.get_stock_ohlc(product)
                    
                    _print(f'Successfully fetched data with shape: {ohlc.shape if hasattr(ohlc, "shape") else len(ohlc)}', 2, verbose)
                    
                    labels_to_ohlc[product] = ohlc.copy(deep=True)
                except Exception as e:
                    _print_error(f'Error fetching price data: {str(e)}')
                    raise ValueError(f'Failed to get price data for {product} : {str(e)}')

            _print(f'Obtained data for product {product} with {len(ohlc)} entries.', 2, verbose)



            

            series = ohlc_to_feature(ohlc, feature, config_feature_calculation)
            series = series.dropna()
            _print(f'Computed feature {feature} with {len(series)} entries.', 1, verbose)

            # Add transformations and normalization
            series.name = ut.get_name_config(config)
            series = add_transformation(series, config_feature)
            series = add_normalization(series, config_feature)
            _print(f'Applied transformations to feature {feature}.', 2, verbose)

            if not series.empty:
                # Add lag if specified
                if 'lag' not in config_feature:
                    concat.append(series)
                else:
                    for lag in config_feature['lag']:
                        series_lag = series.shift(lag)
                        series_lag.name = f'{series.name}_lag_{lag}'
                        concat.append(series_lag)
                        _print(f'Added lag {lag} to feature {feature}.', 2, verbose)
            else:
                concat.append(pd.Series(name=series.name))

        except Exception as e:
            _print_error(e)
            raise ValueError(f'Feature {feature} not available in get_data.')

    _print('All features have been added to the dataframe.', 1, verbose)

    df = pd.concat(concat, axis=1)
    df = ut.sort_date_index(df)
    df = df.dropna()
    
    if return_ohlc:
        return df, labels_to_ohlc
    else:
        return df 
    
def score_to_aggregation(date_to_product_score, method_aggregation, config_aggregation={},product_to_last_date={}):
    """
    This function takes a dictionary where the key is a date and the value is another dictionary
    with the product names as keys and their corresponding scores as values. The function returns
    a new dictionary with the same structure but with the scores converted to signals.

    Parameters
    ----------
    date_to_product_score : dict
        A dictionary where the key is a date and the value is another dictionary with the product
        names as keys and their corresponding scores as values.

    method_aggregation : str
        The method to use to convert the scores to signals. Can be 'sign', 'sign_neg',
        'bottom_top', 'bottom_top_decile', 'long_quantile', or 'repartion_feature'.

    config_aggregation : dict
        A dictionary with the configuration for the aggregation method.

    Returns
    -------
    date_to_product_signal : dict
        A dictionary with the same structure as the input but with the scores converted to signals.
    """

    date_to_product_signal = {}
    dates = list(date_to_product_score.values())
    last_date = dates[-1]
    risky_dates = list(product_to_last_date.values())
    for date, product_to_score in date_to_product_score.items():
        date_without_time = ut.format_datetime(date)
        product_to_score_ = {product : score for product, score in product_to_score.items() if not pd.isna(score)}
        if method_aggregation == 'sign':
            # Convert the scores to signals by taking the sign
            product_to_signal = {product: np.sign(score) for product, score in product_to_score_.items()}
        elif method_aggregation == 'sign_neg':
            # Convert the scores to signals by taking the negative sign
            product_to_signal = {product: -np.sign(score) for product, score in product_to_score_.items()}
        elif method_aggregation == 'bottom_top':
            # Convert the scores to signals by taking the top and bottom products
            bottom = config_aggregation['bottom']
            top = config_aggregation['top']
            reverse = config_aggregation.get('reverse',False)
            product_to_signal = ut.bottom_top_products(product_to_score_, top, bottom,reverse=reverse)
        
        else:
            raise ValueError(f"Method aggregation '{method_aggregation}' not recognized.")
        
        if date == last_date : 
            product_to_signal = {product : 0 for product in product_to_score_.keys()}
        if date_without_time in risky_dates : 
            risky_products = [product for product, product_date in product_to_last_date.items() if product_date == date_without_time]
            print(f'{date_without_time} with products{risky_products}')
            for product in risky_products : 
                product_to_signal[product] = 0
        date_to_product_signal[date] = product_to_signal 

    # Convert dates to string format for JSON serialization
    json_compatible_signals = {}
    json_compatible_scores = {}

    for date, product_scores in date_to_product_score.items():
        json_compatible_scores[date.strftime('%Y-%m-%d %H:%M:%S')] = ut.make_serializable(product_scores)

    for date, product_signals in date_to_product_signal.items():
        json_compatible_signals[date.strftime('%Y-%m-%d %H:%M:%S')] = ut.make_serializable(product_signals)
    
    # Save to JSON files in output directory
    import json
    import os
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save signals
    signals_path = os.path.join(output_dir, 'aggregation_signals.json')
    with open(signals_path, 'w') as f:
        json.dump(json_compatible_signals, f, indent=4)

    # Save scores
    scores_path = os.path.join(output_dir, 'aggregation_scores.json')
    with open(scores_path, 'w') as f:
        json.dump(json_compatible_scores, f, indent=4)

    return date_to_product_signal
    
def add_transformation(series, config_feature):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified transformation applied.
    The transformation is specified in the configuration as a dictionary
    with the transformation name as the key and the configuration for the
    transformation as the value.

    Parameters
    ----------
    series : pandas Series
        The series to apply the transformation to
    config_feature : dict
        The configuration for the feature
        It should contain the following keys:
            transformation : dict
                The dictionary of transformations to apply
                Each key is the name of the transformation and the value is the configuration for the transformation

    Returns
    -------
    pandas Series
        The series with the specified transformation applied
    """
    transformation = config_feature.get('transformation',None)
    if transformation is None : 
        return series
    if series.empty:
        return series
    for transformation_name, config_transformation in transformation.items():
        if transformation_name == 'cross_moving_average':
            series = add_crossing_moving_average(series,config_transformation) 
        elif transformation_name == 'pct': 
            series = add_pct_variation(series,config_transformation) 
        elif transformation_name == 'distance_to_high':
            series = add_distance_to_high(series,config_transformation)
        elif transformation_name == 'distance_to_low':
            series = add_distance_to_low(series,config_transformation) 
        elif transformation_name == 'average':
            series = add_average(series,config_transformation)
        elif transformation_name == 'distance_from_average':
            series = add_distance_from_average(series,config_transformation)
        elif transformation_name == 'vol_ratio':
            series = add_vol_ratio(series,config_transformation)
        elif transformation_name == 'max':
            series = add_max(series,config_transformation)
        elif transformation_name == 'min':
            series = add_min(series,config_transformation) 
        elif transformation_name == 'range_position': 
            series = add_range_position_period(series,config_transformation)
        elif transformation_name == 'z_score':
            series = add_z_score(series,config_transformation) 
        else : 
            raise ValueError(f'The transofromation {transformation_name} not available')
    series.name = f'{series.name}_{transformation_name}'
    return series 

def add_distance_from_average(series,config_transformation):
    period = config_transformation['period']
    series = series.sub(ut.get_moving_average(series,period,'MA'))
    return series 
def add_vol_ratio(series,config_transformation):
    period = config_transformation['period']
    series = series.div(series.rolling(period).std())
    return series 
def add_range_position_period(series,config_transformation):
    period = config_transformation['period']
    series = ut.add_range_position(series,period) 
    return series  
def add_max(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified maximum applied.
    The maximum is calculated over a rolling window of the specified period.

    Parameters
    ----------
    series : pandas Series
        The series to apply the maximum to
    config_transformation : dict
        The configuration for the feature
        It should contain the following keys:
            period : int
                The period to calculate the maximum over

    Returns
    -------
    pandas Series
        The series with the maximum applied
    """
    period = config_transformation['period']
    series = series.rolling(period).max()
    return series 
def add_min(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified minimum applied.
    The minimum is calculated over a rolling window of the specified period.

    Parameters
    ----------
    series : pandas Series
        The series to apply the minimum to
    config_transformation : dict
        The configuration for the transformation
        It should contain the following keys:
            period : int
                The period to calculate the minimum over

    Returns
    -------
    pandas Series
        The series with the minimum applied
    """
    period = config_transformation['period']
    series = series.rolling(period).min()
    return series 
def add_z_score(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified minimum applied.
    The minimum is calculated over a rolling window of the specified period.

    Parameters
    ----------
    series : pandas Series
        The series to apply the minimum to
    config_transformation : dict
        The configuration for the transformation
        It should contain the following keys:
            period : int
                The period to calculate the minimum over

    Returns
    -------
    pandas Series
        The series with the minimum applied
    """
    period = config_transformation['period']
    series_rolling = series.rolling(period)
    series = (series - series_rolling.mean())/series_rolling.std()
    return series
def add_normalization(series, config_feature):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified normalization applied.
    The normalization is either the z-score or None.

    Parameters
    ----------
    series : pandas Series
        The series to apply the normalization to
    config_feature : dict
        The configuration for the feature
        It should contain the following keys:
            normalization : dict
                The configuration for the normalization
                It should contain the following keys:
                    method : str
                        The method to use for the normalization
                        It can be either 'z_score' or None
                    period : int
                        The period to use for the normalization
                        It is required if the method is 'z_score'

    Returns
    -------
    pandas Series
        The normalized Series
    """
    config_normalization = config_feature.get('normalization',None)
    if config_normalization is None : 
        # If there is no normalization, return the original series
        return series 
    if series.empty : 
        return series 
    else : 
        # If there is normalization, extract the method and period
        method = config_normalization['method']
        if method == 'z_score':
            # If the method is z_score, calculate the z_score
            return ut.z_score(series,config_normalization['period'])
        else : 
            # If the method is not z_score, raise a ValueError
            raise ValueError(f'ft.add_normalization. The method {method} not available')
def add_pct_variation(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified transformation applied.
    The transformation is the percentage change.

    Parameters
    ----------
    series : pandas Series
        The series to apply the transformation to
    config_transformation : dict
        The configuration for the transformation
        It should contain the following keys:
            period : int
                The period of the percentage change

    Returns
    -------
    pandas Series
        The transformed Series
    """
    period = config_transformation['period']
    series = series.pct_change(period)
    return series
def add_average(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified transformation applied.
    The transformation is the moving average.

    Parameters
    ----------
    series : pandas Series
        The series to apply the transformation to
    config_transformation : dict
        The configuration for the feature
        It should contain the following keys:
            period : int
                The period of the moving average
            method : str
                The method to use for the moving average
                It can be either 'MA' for the mean or 'EWM' for the exponentially weighted mean
    """
    period = config_transformation['period']
    method = config_transformation['method']
    # Calculate the moving average using the mean or exponentially weighted mean
    series = ut.get_moving_average(series, period, method)
    return series
def add_distance_to_high(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified transformation applied.
    The transformation is the distance to the high.

    Parameters
    ----------
    series : pandas Series
        The series to apply the transformation to
    config_transformation : dict
        The configuration for the transformation
        It should contain the following keys:
            period : int
                The period to calculate the high

    Returns
    -------
    pandas Series
        The transformed Series
    """
    period = config_transformation['period']
    # Calculate the high
    series_high = series.rolling(period).max()
    # Calculate the distance to the high
    series = series.div(series_high) 
    return series 
def add_distance_to_low(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified transformation applied.
    The transformation is the distance to the low.

    Parameters
    ----------
    series : pandas Series
        The series to apply the transformation to
    config_transformation : dict
        The configuration for the feature
        It should contain the following keys:
            period : int
                The period to calculate the low

    Returns
    -------
    pandas Series
        The transformed Series
    """
    period = config_transformation['period']
    series_low = series.rolling(period).min()
    series = series.div(series_low)
    return series
def add_crossing_moving_average(series, config_transformation):
    """
    This function takes a pandas Series and a configuration for a feature
    and returns a new Series with the specified transformation applied.
    The transformation is a crossing moving average.
    
    Parameters
    ----------
    series : pandas Series
        The series to apply the transformation to
    config_feature : dict
        The configuration for the feature
        It should contain the following keys:
            means : list
                The list of means to use for the moving average
            method : str
                The method to use for the moving average
                It can be either 'MA' or 'EWM'
    
    Returns
    -------
    pandas Series
        The transformed Series
    """
    
    # Get the list of means from the configuration
    means = config_transformation['means']
    
    # Get the method to use for the moving average from the configuration
    method = config_transformation['method']
    
    # If there is only one mean, subtract the moving average from the series
    if len(means) == 1:
        series -= ut.get_moving_average(series, means[0], method)
        
    # If there are two means, subtract the long moving average from the short moving average
    elif len(means) == 2:
        # Get the short and long means
        short_mean = min(means)
        long_mean = max(means)
        
        # Calculate the short and long moving averages
        short_moving_average = ut.get_moving_average(series, short_mean, method)
        long_moving_average = ut.get_moving_average(series, long_mean, method)
        
        # Subtract the long moving average from the short moving average
        series = short_moving_average - long_moving_average
        
    # If there are more than two means, raise a ValueError
    else:
        raise ValueError(f'There is no crossing moving average with means {means}')
        
    # Return the transformed Series
    return series