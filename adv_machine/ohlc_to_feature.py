import pandas as pd 
import numpy as np 

from datetime import datetime, timedelta 
from sklearn.linear_model import LinearRegression 
from hurst import compute_Hc 
from scipy import stats 

import adv_machine.get_prices as gp 
import adv_machine.utils as ut 
from adv_machine.log import _print,_print_error    


def ohlc_to_feature(ohlc, feature, config_feature, verbose=0):
    _print(f"Computing feature: {feature} with config: {config_feature}", 2, verbose)
    
    if feature == 'average_max_return':
        series = add_average_max_returns(ohlc, config_feature)
    elif feature == 'beta':
        series = add_beta(ohlc,config_feature)
    elif feature == 'alpha':
        series = add_alpha(ohlc,config_feature)
    elif feature == 'average_min_return': 
         series = add_average_min_returns(ohlc,config_feature)
    elif feature == 'calmar_ratio':
        series = add_calmar_ratio(ohlc, config_feature)
    elif feature == 'close':
        series = ohlc['close']
    elif feature == 'volume':
         series = ohlc['volume']
    elif feature == 'market_cap':
         series = ohlc['market_cap']
    elif feature == 'distance_from_vwap':
        series = add_distance_from_vwap(ohlc,config_feature) 
    elif feature == 'distance_from_high':
        series = add_distance_from_high(ohlc,config_feature)
    elif feature == 'distance_from_low':
        series = add_distance_from_low(ohlc,config_feature)
    elif feature == 'distance_from_low_vol_ratio':
        series = add_distance_from_low_vol_ratio(ohlc,config_feature)
    elif feature == 'distance_from_ma':
        series = add_distance_from_ma(ohlc,config_feature)
    elif feature == 'distance_from_ewma':
        series = add_distance_from_ewma(ohlc,config_feature)
    elif feature == 'distance_from_ma_volume':
        series = add_distance_from_ma_volume(ohlc,config_feature)
    elif feature == 'excess_return':
        series = add_excess_return(ohlc,config_feature)
    elif feature == 'excess_return_weight_marketcap':
        series = add_excess_return_weight_marketcap(ohlc,config_feature)
    elif feature == 'count_excess_return':
        series = add_count_excess_return(ohlc,config_feature)
    elif feature == 'high':
        series = ohlc['high'] 
    elif feature == 'hurst': 
        series = add_hurst(ohlc,config_feature)
    elif feature == 'illiquid_ratio':
        series = add_illiquid_ratio(ohlc,config_feature)
    elif feature == 'lag_log_price':
        series = add_lag_log_price(ohlc,config_feature)
    elif feature == 'low':
        series = ohlc['low']
    elif feature == 'open':
        series = ohlc['open']
    elif feature == 'price_path_convexity':
        series = add_price_path_convexity(ohlc, config_feature)
    elif feature == 'price_volume_ratio':
        series = add_price_volume_ratio(ohlc, config_feature)
    elif feature == 'r':
        series = add_r(ohlc,config_feature)
    elif feature == 'r_adjusted_return':
        series = add_r_adjusted_return(ohlc,config_feature)
    elif feature == 'r_adjusted_range_position':
        series = add_r_adjusted_range_position(ohlc,config_feature)
    elif feature == 'range_position':
        series = add_range_position(ohlc, config_feature)
    elif feature == 'realized_volatility':
        series = add_yang_zhang_volatility(ohlc, config_feature)
    elif feature == 'return_pct':
        series = add_returns(ohlc, config_feature)
    elif feature == 'return_vol_ratio':
        series = add_return_vol_ratio(ohlc, config_feature)
    elif feature == 'skew':
        series = add_skew(ohlc,config_feature)
    elif feature == 'spearman':
        series = add_spearman(ohlc,config_feature)
    elif feature == 'spearman_adjusted_range_position':
        series = add_spearman_adjusted_range_position(ohlc,config_feature)
    elif feature == 'volume':
        series = ohlc['volume'] 
    elif feature == 'volume_imbalance':
        series = add_volume_imbalance(ohlc,config_feature)
    elif feature == 'volume_pct':
        series = add_volume_pct(ohlc, config_feature)
    elif feature == 'volume_range_position':
        series = add_volume_range_position(ohlc, config_feature)
    elif feature == 'distance_from_ewma_volume':
        series = add_distance_from_ewma_volume(ohlc, config_feature)
    elif feature == 'volume_weighted_return':
        series = add_volume_weighted_return(ohlc, config_feature) 
    elif feature == 'vwap':
        series = add_vwap(ohlc,config_feature) 
    elif feature == 'var':
         series = add_var(ohlc,config_feature)
    else:
        error_msg = f'The feature {feature} is not available'
        _print_error(error_msg)
        raise ValueError(error_msg)
    series = series.dropna()
    
    # Build feature name from config
    result = []
    for key, value in config_feature.items():
        result.append(f"{key}_{value}")
    config_feature_name = '_'.join(result)
    
    # Set the series name using the properly formatted string
    series.name = f'{feature}_{config_feature_name}'
    
    _print(f"Successfully computed {feature} feature with shape {series.shape}", 3, verbose)
    return series 

def add_average_max_returns(ohlc, config_feature):
    period_return = config_feature['period_return']
    period_lookback = config_feature['period_lookback']
    count_max = config_feature['count_max']
    
    # Calculate the percentage change for the specified return period
    series_return = ohlc['close'].pct_change(period_return)
    
    # Calculate the rolling window of the specified lookback period
    rolling_max = series_return.rolling(window=period_lookback)
    
    # Apply a function to get the mean of the top 'count_max' returns in each window
    series = rolling_max.apply(lambda x: np.mean(np.sort(x)[-count_max:]), raw=True)
def add_beta(ohlc, config_feature):
    period = config_feature['period']
    market_name = '^GSPC'
    ohlc_market = gp.get_stock_ohlc(market_name)
    series_market_return = ohlc_market['close'].pct_change(period)
    series_coin_return = ohlc['close'].pct_change(period)
    df = pd.concat([series_coin_return, series_market_return], axis=1)
    df = ut.sort_date_index(df)
    df = df.dropna()
    df.columns = ['coin', 'market']

    # Calculate rolling beta using covariance and variance
    series = df.rolling(period).apply(
        lambda x: np.cov(x['coin'], x['market'])[0,1] / np.var(x['market'])
        if len(x) > 1 else np.nan
    )
    
    return series 
def add_alpha(ohlc, config_feature):
    period = config_feature['period']
    market_name = '^GSPC'
    ohlc_market = gp.get_stock_ohlc(market_name)
    series_market_return = ohlc_market['close'].pct_change(period)
    series_coin_return = ohlc['close'].pct_change(period)
    df = pd.concat([series_coin_return, series_market_return], axis=1)
    df = ut.sort_date_index(df)
    df = df.dropna()
    df.columns = ['coin', 'market']

    # Calculate rolling alpha (intercept) using mean returns and beta
    series = df.rolling(period).apply(
        lambda x: x['coin'].mean() - (np.cov(x['coin'], x['market'])[0,1] / np.var(x['market'])) * x['market'].mean()
        if len(x) > 1 else np.nan
    )
    
    return series 
    
 
def add_average_min_returns(ohlc, config_feature):
    period_return = config_feature['period_return']
    period_lookback = config_feature['period_lookback']
    count_min = config_feature['count_min']
    
    # Calculate the percentage change for the specified return period
    series_return = ohlc['close'].pct_change(period_return)
    
    # Calculate the rolling window of the specified lookback period
    rolling_max = series_return.rolling(window=period_lookback)
    
    # Apply a function to get the mean of the top 'count_max' returns in each window
    series = rolling_max.apply(lambda x: np.mean(np.sort(x)[:count_min]), raw=True)
    
    return series 
def add_lag_log_price(ohlc,config_feature):
    period = config_feature['period']
    series = np.log(ohlc['close']).shift(period)
    return series 
 
 

def add_calmar_ratio(ohlc, config_feature):
    def calmar_ratio(window):
        def calculate_max_drawdown(window):
            cum_returns = (1 + window).cumprod()
            peak = cum_returns.iloc[0]
            max_drawdown = 0
            for val in cum_returns:
                if val > peak:
                    peak = val
                else:
                    drawdown = (peak - val) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            return max_drawdown

        cum_returns = (1 + window).prod() - 1
        max_drawdown = calculate_max_drawdown(window)
        return cum_returns / max_drawdown
    period = config_feature["period"]
    series = ohlc.close.pct_change().rolling(period).apply(calmar_ratio)
    return series 
def add_illiquid_ratio(ohlc,config_feature):
    period = config_feature['period']
    series = np.abs(ohlc['close'].pct_change()).mean()/ohlc['volume'].rolling(period).mean()
    return series 

def add_distance_from_high(ohlc, config_feature):
        period           = config_feature["period"]
        series = ohlc.close / ohlc.high.rolling(period).max() - 1
        return series 
def add_excess_return(ohlc,config_feature):
     period_return= config_feature['period_return']
     period_lookback = config_feature['period_lookback']
     market = 'BTCUSDT'
     series_market = gp.get_prices(market,'future',drive=False)['close']
     series_return_market = series_market.pct_change(period_return)
     series_return_coin = ohlc['close'].pct_change(period_return)
     df = pd.concat([series_return_coin,series_return_market],axis=1)
     df.columns = ['coin','market']
     df = ut.sort_date_index(df)
     df = df.dropna()
     series = df['coin'] - df['market']
     series = series.ewm(span=period_lookback)
     return series 
def add_excess_return_weight_marketcap(ohlc,config_feature):
     period_return= config_feature['period_return']
     period_lookback = config_feature['period_lookback']
     market = 'BTCUSDT'
     ohlc_market = gp.get_prices(market,'future',drive=False)
     series_return_market = ohlc_market['close'].pct_change(period_return)
     series_marketcap_market = ohlc_market['volume']
     series_marketcap_coin = ohlc['volume']
     series_return_coin = ohlc['close'].pct_change(period_return)
     df = pd.concat([series_return_coin,series_return_market,series_marketcap_coin,series_marketcap_market],axis=1)
     df.columns = ['coin','market','marketcap_coin','marketcap_market']
     df = ut.sort_date_index(df)
     df = df.dropna()
     series = df['coin'] - df['market']
     series = series * df['marketcap_coin'] / df['marketcap_market']
     series = series.ewm(span=period_lookback)
     return series 

def add_count_excess_return(ohlc,config_feature):
     period_return = config_feature['period_return']
     period_lookback = config_feature['period_lookback']
     market = 'BTCUSDT'
     series_market = gp.get_prices(market,'future',drive=False)['close']
     series_return_market = series_market.pct_change(period_return)
     series_return_coin = ohlc['close'].pct_change(period_return)
     df = pd.concat([series_return_coin,series_return_market],axis=1)
     df.columns = ['coin','market']
     df = ut.sort_date_index(df)
     df = df.dropna()
     series = df['coin'] - df['market']
     series = series.rolling(period_lookback).apply(lambda x : np.sum([1 for excess_return in x if excess_return > 0])/period_lookback)
     return series 
      

def add_distance_from_ma(ohlc, config_feature):
        period           = config_feature["period"]
        series = ohlc.close / ohlc.close.rolling(period).mean() - 1
        return series

def add_distance_from_ewma(ohlc, config_feature):
        period           = config_feature["period"]
        series = ohlc.close / ohlc.close.ewm(span=period).mean() - 1
        return series

def add_distance_from_ma_volume( ohlc, config_feature):
        period           = config_feature["period"]
        series = ohlc.volume / ohlc.volume.rolling(period).mean() - 1
        return series 

def add_price_path_convexity(ohlc,config_feature):
    period = config_feature['period']
    series = ohlc['close']
    series = series.rolling(period).apply(lambda x : (np.mean(x) +(1/2)*(x[0]+x[-1]))/np.mean(x))
    return series 

def add_hurst(ohlc, config_feature):
        period = config_feature["period"]
        series = ohlc.close.rolling(period).apply(lambda x: compute_Hc(x)[0])
        return series 

def add_price_volume_ratio(ohlc,config_feature):
    series = ohlc['close']/ohlc['volume']
    return series 
def add_skew(ohlc, config_feature):
        period = config_feature["period"]
        series = ohlc.close.pct_change().rolling(period).skew()
        return series 

def add_range_position(ohlc,config_feature):
    period = config_feature["period"]
    series = ut.add_range_position(ohlc["close"],period)
    return series 

def add_spearman(ohlc, config_feature):
        period       = config_feature["period"]
        series = ohlc.close.rolling(period).apply(
            lambda x: stats.spearmanr(x.values, np.arange(period)).statistic
        )
        return series

def add_spearman_adjusted_range_position(ohlc, config_feature):
        period           =  config_feature["period"]
        series           =  ohlc.close.rolling(period).apply(
            lambda x: abs(stats.spearmanr(x.values, np.arange(period)).statistic) * np.interp(x.iloc[-1], [x.min(), x.max()], [-1, 1])
        )
        return series 

def add_r(ohlc, config_feature):
        period       = config_feature["period"]
        series       = ohlc.close.rolling(period).apply(
            lambda x: np.corrcoef(x.values, np.arange(period))[0][1]
        )
        return series 

def add_r_adjusted_return(ohlc, config_feature):
        period       = config_feature["period"]
        series = ohlc.close.rolling(period).apply(
            lambda x: abs(np.corrcoef(x.values, np.arange(period))[0][1]) * (x.iloc[-1] / x.iloc[0] - 1)
        )
        return series 

def add_spearman_adjusted_return(ohlc, config_feature):
        period       = config_feature["period"]
        series       =       ohlc.close.rolling(period).apply(
            lambda x: abs(stats.spearmanr(x.values, np.arange(period)).statistic) * (x.iloc[-1] / x.iloc[0] - 1)
        )
        return series 
def add_r_adjusted_range_position(ohlc, config_feature):
        period       = config_feature["period"]
        series = ohlc.close.rolling(period).apply(
            lambda x: abs(np.corrcoef(x.values, np.arange(period))[0][1]) * np.interp(x.iloc[-1], [x.min(), x.max()], [-1, 1])
        )
        return series 
def add_distance_from_low(ohlc, config_feature):
        period           = config_feature["period"]
        series = ohlc.close / ohlc.low.rolling(period).min() - 1
        return series 

def add_distance_from_low_vol_ratio(ohlc, config_feature):
        period           = config_feature["period"]
        series_low = ohlc.close / ohlc.low.rolling(period).min() - 1
        series = series_low/series_low.rolling(period).std()
        return series

def add_return_vol_ratio(ohlc, config_feature):
    period = config_feature["period"]
    pct_change = ohlc.close.pct_change()
    rolling_window = pct_change.rolling(period)
    mean = rolling_window.mean()
    vol = rolling_window.std()
    series = mean / vol
    return series

def add_returns(ohlc,config_feature):
    period = config_feature["period"]
    series = ohlc.close.pct_change(period)
    return series   

def add_volume_weighted_return(ohlc,config_feature):
    period = config_feature['period']
    series_return = ohlc.close.pct_change(period=period)
    series_volume = ohlc.volume.ewm(span=period).mean()
    series = series_return*series_volume 
    return series 

def add_yang_zhang_volatility(ohlc,config_feature):
    period = config_feature['period'] 
    log_ho = (ohlc['high'] / ohlc['open']).apply(np.log)
    log_lo = (ohlc['low'] / ohlc['open']).apply(np.log)
    log_co = (ohlc['close'] / ohlc['open']).apply(np.log)
    
    log_oc = (ohlc['open'] / ohlc['close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (ohlc['close'] / ohlc['close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(window=period,center=False).sum() * (1.0 / (period - 1.0))
    open_vol = log_oc_sq.rolling(window=period,center=False).sum() * (1.0 / (period - 1.0))
    window_rs = rs.rolling(window=period,center=False).sum() * (1.0 / (period - 1.0))

    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(365)

    return result.dropna()
def add_volume_imbalance(ohlc, config_feature):
        period = config_feature["period"]
        series = ohlc.buy_volume.rolling(period).sum() / ohlc.volume.rolling(period).sum() - 0.5
        return series 
def add_volume_pct(ohlc, config_feature):
        period = config_feature["period"]
        series = ohlc.volume.pct_change(periods=period)
        return series 
def add_volume_range_position(ohlc, config_feature):
        period = config_feature["period"]
        series = ut.add_range_position(ohlc['volume'],period)
        return series 
def add_distance_from_ewma_volume(ohlc, config_feature):
        period = config_feature["period"]
        series = ohlc['volume'] / ohlc['volume'].ewm(span=period).mean() - 1
        return series 

def add_vwap(ohlc, config_feature):
        period           = config_feature["period"]
        series = ohlc.volume.rolling(period).sum() / ohlc.volume.rolling(period).sum()
        return series

def add_distance_from_vwap(ohlc, config_feature):
        period           = config_feature["period"]
        vwap             = (ohlc.close * ohlc.volume).rolling(period).sum() / ohlc.volume.rolling(period).sum()
        series = ohlc.close / vwap - 1
        return series 
def add_var(ohlc, config_feature):
    period = config_feature['period']
    # Calculate rolling 5th percentile of returns
    series = ohlc['close'].pct_change().rolling(period).quantile(0.05)
    return series 
 

def convert_freq(input,freq,timestamps=None):
    '''
    The input need to have minute timestamp 
    
    '''
    if freq == '1m':
        if (timestamps is None)  :
            df = input.copy(deep=True) 
        elif isinstance(timestamps,list):
            df = ut.get_some_timestamp_index(input,timestamps) 
    elif freq == '1d':
        if (timestamps is None) or (len(set(timestamps)) > 1 ) or (timestamps == []):
            raise ValueError(f'gd.convert_freq. The freq is daily but timestamps {timestamps} was given')
        else :
            timestamp = timestamps[0]
            df = ut.index_to_closest_timeframe_low(input,timestamp) 
    elif freq == 'keep':
        df = input 
    df = ut.sort_date_index(df)
    return df 

def get_beta(ticker,market_coins,period,timestamp):
    config_coin = ut.get_coin_config(ticker)
    ohlc_coin = gp.get_prices(**config_coin)
    series_coin = convert_freq(ohlc_coin['close'],'1d',timestamp)
    series_market = get_market_return(market_coins,config_coin['trading_type'],period = '1d',timestamp=timestamp)
    
    df = pd.concat([series_coin,series_market],axis=1).dropna()
    df = ut.sort_date_index(df).dropna()
    df.columns = ['coin','market']
    dates = []
    beta = []
    
    for df_ in df.rolling(period):
        date_ = df_.index[-1]
        beta_ = np.corrcoef(df_['coin'].values,df_['market'].values)[0][1]
        dates.append(date_)
        beta.append(beta_)
    series_beta = pd.Series(beta,index = dates)
    
    return series_beta 

def get_iv(ticker,market_coins,period,timestamp):
    config_coin = ut.get_coin_config(ticker)
    ohlc_coin = gp.get_prices(**config_coin)
    series_coin = convert_freq(ohlc_coin['close'],'1d',timestamp)
    series_market = get_market_return(market_coins,config_coin['trading_type'],period = '1d',timestamp=timestamp)
    
    df = pd.concat([series_coin,series_market],axis=1).dropna()
    df = ut.sort_date_index(df).dropna()
    df.columns = ['coin','market']
    dates = []
    iv = []
    
    for df_ in df.rolling(period):
        if len(df_) > period-1:
            date_ = df_.index[-1]
            model = LinearRegression(fit_intercept=False)
            model.fit(df_['market'],df_['coin'])
            pred = model.predict(df_['market'])
            resid = df_['coin'][-1] - pred[-1]
            iv = iv.append(resid)
            dates.append(date_)
    series_beta = pd.Series(iv,index = dates)
    
    return series_beta 

def get_market_return(market_coins, trading_type,period,timestamp=None):
    concat = []
    for coin in market_coins  : 
        ohlc = gp.get_prices(coin,trading_type)
        series_return = ohlc['close'].copy(deep=True)
        if 'd' in period : 
            series_return = convert_freq(series_return,'1d',timestamp)
        elif 'm' in period : 
            series_return = series_return
        else : 
            raise ValueError(f'gd.get_market_return. period {period} not available')
        period_ = ut.extract_integers(period)[0]
        series_return = series_return.pct_change(period_)
        concat.append(series_return)
    df = pd.concat(concat,axis=1)
    series = df.mean(axis=1).dropna()
    return series 

def get_streak(ticker,market_coins,timestamp):
    """
    Calculate the streak of consecutive days where the cryptocurrency outperforms
    or underperforms the market based on daily returns.
    
    Parameters:
    - crypto_returns: List or pandas Series of daily returns for the cryptocurrency.
    - market_returns: List or pandas Series of daily returns for the market.
    
    Returns:
    - streaks: List or pandas Series of streak scores.
    """
    # Check if inputs are pandas Series; if not, convert them to Series
    config_coin = ut.get_coin_config(ticker)
    series_coin_return = gp.get_prices(**config_coin)
    series_coin_return = convert_freq(series_coin_return,'1d',timestamp)
    series_market_return = get_market_return(market_coins,config_coin['trading_type'],period=1,timestamp=timestamp)

    df = pd.concat([series_coin_return,series_market_return],axis=1)
    df.columns = ['coin','market']
    df = ut.sort_date_index(df)
    df = df.dropna()

    # Initialize the streaks list
    streaks = [0]  # The first day has no streak, so it is 0

    # Initialize the current streak
    current_streak = 0
    crypto_returns = df['coin']
    market_returns = df['market']
    dates = []

    # Loop through the returns from the first day onward
    for i in range(len(df)):
        if crypto_returns[i] > market_returns[i]:  # Crypto outperforms the market
            if current_streak >= 0:
                current_streak += 1  # Continue the winning streak
            else:
                current_streak = 1  # Reset to a new winning streak
        elif crypto_returns[i] < market_returns[i]:  # Crypto underperforms the market
            if current_streak <= 0:
                current_streak -= 1  # Continue the losing streak
            else:
                current_streak = -1  # Reset to a new losing streak
        else:
            current_streak = 0  # No streak if returns are equal
        
        # Append the current streak to the streaks list
        date = df.index[i]
        dates.append(date)
        streaks.append(current_streak)

    return pd.Series(streaks,index = dates)    

 