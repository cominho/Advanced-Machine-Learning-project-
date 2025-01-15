import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta 

import adv_machine.utils as ut 


def get_return_per_date_report(series_cumulative_pnl, initial_capital): 
    df = pd.DataFrame(series_cumulative_pnl.values, index=series_cumulative_pnl.index, columns=['cumulative_pnl'])
    df['return'] = ut.compute_returns_from_cumulative_pnl(df['cumulative_pnl'], initial_capital) 
    df['year_month'] = [x.strftime('%Y-%m') for x in df.index]
    df['year'] = [x.strftime('%Y') for x in df.index]
    df['month'] = [x.strftime('%m') for x in df.index]
    
    df_return_per_month = df.groupby('year_month')['cumulative_pnl'].apply(lambda x: ((initial_capital + x.iloc[-1]) / (initial_capital + x.iloc[0]) - 1)*100).reset_index()
    df_return_per_month.columns = ['year_month', 'return_per_month']
    df_return_per_month['year'] = df_return_per_month['year_month'].apply(lambda x : datetime.strptime(x,'%Y-%m').strftime('%Y'))
    df_return_per_month['month'] = df_return_per_month['year_month'].apply(lambda x : datetime.strptime(x,'%Y-%m').strftime('%m')) 
    
    # Calculate volatility per annum
    df_volatility = df.groupby('year')['return'].std().reset_index()
    df_volatility.columns = ['year', 'volatility_per_annum']
    
    # Calculate annual return
    df_annual_return = df.groupby('year')['cumulative_pnl'].apply(lambda x: 100 * ((initial_capital + x.iloc[-1]) / (initial_capital + x.iloc[0]) - 1)).reset_index()
    df_annual_return.columns = ['year', 'annual_return']
    
    # Calculate sharpe per annum
    df_sharpe = df.groupby('year')['return'].apply(lambda x: (x.mean() / x.std()) * np.sqrt(365)).reset_index()
    df_sharpe.columns = ['year', 'sharpe_per_annum']
    
    df_pivot = df_return_per_month.pivot(index='year', columns='month', values='return_per_month')
    df_pivot.columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] 

    # Ensure 'year' column is of the same type in all DataFrames
    df_pivot.index = df_pivot.index.astype(str)
    df_volatility['year'] = df_volatility['year'].astype(str)
    df_annual_return['year'] = df_annual_return['year'].astype(str)
    df_sharpe['year'] = df_sharpe['year'].astype(str)
    
    # Merge the volatility, annual return, and sharpe data
    df_pivot = df_pivot.merge(df_volatility, on='year', how='left')
    df_pivot = df_pivot.merge(df_annual_return, on='year', how='left')
    df_pivot = df_pivot.merge(df_sharpe, on='year', how='left')
    
    # Convert all columns to numeric, coercing errors to NaN 
    df_pivot = df_pivot.apply(pd.to_numeric, errors='coerce') 
    df_pivot.index = df_pivot.year 
    df_pivot.drop('year',axis=1,inplace=True) 
    
    
    
    return df_pivot 

def get_metric_per_quantile_portfolio(quantile_to_backtest):
    quantile_to_metrics = {}
    for quantile, result in quantile_to_backtest.items():
        cumulative_pnl = result['cumulative_pnl'].iloc[-1]
        sharpe_ratio = result['sharpe_ratio']
        date_to_product_signal = result['date_to_signal']
        size = np.mean([len([product for product , signal in product_signal.items() if np.abs(signal)>0]) for date,product_signal in date_to_product_signal.items()])
        quantile_to_metrics[quantile] = {'cumulative_pnl' : cumulative_pnl,'sharpe_ratio':sharpe_ratio,'size':size}
    df_quantile_metrics = pd.DataFrame(list(quantile_to_metrics.values()),index = list(quantile_to_metrics.keys()))
    return df_quantile_metrics 

def get_pnl_last_lookback_month(series_cumulative_pnl, lookback_months=1):
    # Determine the end date of the series
    end_date = series_cumulative_pnl.index[-1]
    
    # Calculate the start date for the lookback period
    start_date = end_date - pd.DateOffset(months=lookback_months)
    
    # Filter the series to include only the last `lookback_months` period
    lookback_pnl = series_cumulative_pnl.loc[start_date:end_date]
    
    # Calculate total months in the series
    total_months = (series_cumulative_pnl.index[-1] - series_cumulative_pnl.index[0]).days / 30.44  # average days per month
    
    # Calculate the PnL for the lookback period
    if len(lookback_pnl) > 1:  # Need at least 2 points to calculate PnL
        pnl_last_lookback = lookback_pnl.iloc[-1] - lookback_pnl.iloc[0]
        return pnl_last_lookback
    else:
        print(f"Not enough data for {lookback_months} month(s) lookback period. Series only contains {round(total_months, 1)} months of data.")
        return np.nan

def get_sharpe_ratio_lookback_month(series_returns, lookback):
    # Filter the series to include only the last `lookback` month
    end_date = series_returns.index[-1]
    start_date = end_date - pd.DateOffset(months=lookback)
    lookback_returns = series_returns.loc[start_date:end_date]

    # Calculate total months in the series
    total_months = (series_returns.index[-1] - series_returns.index[0]).days / 30.44  # average days per month

    # Check if we have enough data
    if len(lookback_returns) <= 1:
        print(f"Not enough data for {lookback} month(s) lookback period. Series only contains {round(total_months, 1)} months of data.")
        return np.nan

    # Calculate mean and standard deviation of the lookback returns
    mean_return = np.mean(lookback_returns)
    std_return = np.std(lookback_returns)

    # Check for zero standard deviation
    if std_return == 0:
        print(f"Standard deviation of returns is zero, cannot calculate Sharpe ratio for lookback period")
        return np.nan

    # Calculate the Sharpe ratio
    sharpe_ratio = np.sqrt(365 / len(lookback_returns)) * mean_return / std_return
    return np.round(sharpe_ratio, 2)

def get_trades_amount(date_to_product_signal):
    long_amount = 0
    short_amount = 0
    for date , product_to_signal in date_to_product_signal.items():
        for product, signal in product_to_signal.items():
            if signal > 0 : 
                long_amount = long_amount + 1 
            elif signal < 0 : 
                short_amount = short_amount +1 
            else : 
                pass 
    return long_amount, short_amount 

def create_product_signal_probabilities_df(date_to_product_signal):
    """
    Calculate the probability for each product to be long or short based on the date_to_product_signal dictionary.
    
    :param date_to_product_signal: Dictionary where the key is a date (string) and the value is a dictionary of products and their signals.
    :return: DataFrame with index as product names and columns 'Long' and 'Short' representing the probabilities.
    """
    # Initialize a dictionary to store counts for each product
    product_signal_counts = {}
    product_to_number_of_dates = {}

    # Iterate over the date_to_product_signal to count long and short signals for each product
    for date, product_signals in date_to_product_signal.items():
        for product, signal in product_signals.items():
            if product not in product_signal_counts:
                product_signal_counts[product] = {'Long': 0, 'Short': 0, 'Neutral': 0}
                product_to_number_of_dates[product] = 1
            if signal > 0:
                product_signal_counts[product]['Long'] += 1
            elif signal < 0:
                product_signal_counts[product]['Short'] += 1  
            else:
                product_signal_counts[product]['Neutral'] += 1 
            product_to_number_of_dates[product] += 1

    # Calculate the total number of dates
    total_dates = len(date_to_product_signal)

    # Convert counts to probabilities
    for product in product_signal_counts:
        number_dates = product_to_number_of_dates[product]
        product_signal_counts[product]['Long'] /= number_dates
        product_signal_counts[product]['Short'] /= number_dates 
        product_signal_counts[product]['Neutral'] /= number_dates 

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(product_signal_counts, orient='index').reset_index()
    df.columns = ['product', 'Long', 'Short','Neutral']
    df = df.sort_values(by = 'Long') 
    
    return df 

def create_product_signal_transition_probabilities_df(date_to_product_signal):
    """
    Calculate the transition probabilities for each product based on the date_to_product_signal dictionary.
    
    :param date_to_product_signal: Dictionary where the key is a date (string) and the value is a dictionary of products and their signals.
    :return: DataFrame with index as product names and columns representing the transition probabilities.
    """
    # Initialize dictionaries
    product_transition_counts = {}
    product_to_last_position = {}
    dates = list(date_to_product_signal.keys())
    product_to_number_of_dates = {}

    # Iterate over the dates to count transitions
    for date in dates:
        product_to_signal_curr_date = date_to_product_signal[date]

        for product, curr_signal in product_to_signal_curr_date.items():
            if product not in product_transition_counts:
                product_transition_counts[product] = {'Neutral_to_Short':0,'Neutral_to_Long':0,'Long_to_Neutral':0,'Short_to_Neutral':0,'Long_to_Short': 0, 'Short_to_Long': 0, 'Short_to_Short': 0, 'Long_to_Long': 0}
                product_to_number_of_dates[product] = 1 
                product_to_last_position[product] = curr_signal 
            else:
                prev_signal = product_to_last_position[product]
                product_to_last_position[product] = curr_signal
                product_to_number_of_dates[product] += 1

                if prev_signal > 0 and curr_signal < 0:
                    product_transition_counts[product]['Long_to_Short'] += 1
                elif prev_signal < 0 and curr_signal > 0:
                    product_transition_counts[product]['Short_to_Long'] += 1
                elif prev_signal < 0 and curr_signal < 0:
                    product_transition_counts[product]['Short_to_Short'] += 1
                elif prev_signal > 0 and curr_signal > 0:
                    product_transition_counts[product]['Long_to_Long'] +=1 
                elif prev_signal > 0 and curr_signal == 0:
                    product_transition_counts[product]['Long_to_Neutral'] += 1
                elif prev_signal < 0 and curr_signal == 0:
                    product_transition_counts[product]['Short_to_Neutral'] +=1
                elif prev_signal == 0 and curr_signal > 0:
                    product_transition_counts[product]['Neutral_to_Long'] +=1 
                elif prev_signal == 0 and curr_signal < 0:
                    product_transition_counts[product]['Neutral_to_Short'] += 1 

    # Calculate the total number of transitions for each product
    for product in product_transition_counts:
        total_transitions = product_to_number_of_dates[product]
        if total_transitions > 0:
            for transition in product_transition_counts[product]:
                product_transition_counts[product][transition] /= total_transitions

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(product_transition_counts, orient='index').reset_index()
    df.columns = ['product', 'Neutral_to_Short','Neutral_to_Long','Long_to_Neutral','Short_to_Neutral','Long_to_Short', 'Short_to_Long', 'Short_to_Short', 'Long_to_Long']
    
    return df 

def calculate_drawdowns(series_cumulative_pnl):
    """
    Calculate drawdowns, percentage down, and drawdown duration.
    
    Args:
    series_cumulative_pnl (pd.Series): Series of cumulative PnL values.
    
    Returns:
    pd.DataFrame: DataFrame containing drawdown metrics.
    """
    # Initialize variables 
    series_cumulative_pnl_drawdown = series_cumulative_pnl.copy(deep=True)
    drawdown_data = []
    peak = series_cumulative_pnl_drawdown.iloc[0]
    drawdown_start = series_cumulative_pnl_drawdown.index[0]
    in_drawdown = False

    for date, value in series_cumulative_pnl_drawdown.items():
        if value >= peak:
            if in_drawdown:
                # If in a drawdown, this is the recovery point
                in_drawdown = False
            peak = value
            drawdown_start = date
        else:
            if not in_drawdown:
                # Start of a new drawdown
                drawdown_start = date
                drawdown_data.append({
                    'Start Date': drawdown_start,
                    'Start Value': series_cumulative_pnl_drawdown[drawdown_start]
                })
                in_drawdown = True

            # Update the current drawdown details
            trough_value = value
            trough_date = date
            drawdown_data[-1]['Trough Date'] = trough_date
            drawdown_data[-1]['Trough Value'] = trough_value
            drawdown_data[-1]['Drawdown Duration'] = (trough_date - drawdown_start).days
            drawdown_data[-1]['Percentage Drawdown'] = (
                (drawdown_data[-1]['Start Value'] - trough_value) / drawdown_data[-1]['Start Value'] * 100
                if drawdown_data[-1]['Start Value'] != 0 else 0
            )

    # Create DataFrame
    drawdown_df = pd.DataFrame(drawdown_data)
    drawdown_df = drawdown_df.sort_values(by='Percentage Drawdown', ascending=False)
    drawdown_df = drawdown_df.reset_index(drop=True)
    
    return drawdown_df 

def get_metric_to_value(backtest_report, lookback_metrics=[12, 24]):
    """
    Calculate various metrics from a backtest report and return them in a dictionary.
    
    Args:
        backtest_report (dict): Dictionary containing backtest results with at least 'cumulative_pnl' and 'returns'
        lookback_metrics (list): List of lookback periods in months to calculate metrics for
    
    Returns:
        dict: Dictionary containing calculated metric values
    """
    metric_to_value = {}
    
    # Calculate lookback metrics
    for lookback in lookback_metrics:
        # Calculate PnL for lookback period
        cumulative_pnl_lookback = get_pnl_last_lookback_month(
            backtest_report["cumulative_pnl"],
            lookback
        )
        
        # Calculate Sharpe ratio for lookback period
        sharpe_ratio_lookback = get_sharpe_ratio_lookback_month(
            backtest_report["returns"],
            lookback
        )
        
        # Store metrics in dictionary
        metric_to_value[f'sharpe_ratio_last_{lookback}_months'] = np.round(sharpe_ratio_lookback, 2)
        metric_to_value[f'cumulative_pnl_last_{lookback}_months'] = np.round(cumulative_pnl_lookback, 0)
    
    # Calculate overall strategy metrics
    metric_to_value['sharpe_ratio_strat'] = np.round(
        get_sharpe_ratio(backtest_report['returns']),
        2
    )
    metric_to_value['cumulative_pnl_strat'] = np.round(
        backtest_report['cumulative_pnl'].iloc[-1],
        0
    )
    
    return metric_to_value 

def get_profit_factor(series_cumulative_pnl):
    series_pnl = series_cumulative_pnl.diff()
    wins = series_pnl[series_pnl > 0].sum()
    losses = series_pnl[series_pnl < 0].sum()
    profit_factor = wins/losses 
    return round(profit_factor,2) 

def get_sharpe_ratio(series_returns, execution_period=1):
    try:
        mean_return = np.mean(series_returns)
        std_return = np.std(series_returns)
        if std_return == 0:
            print(f"Standard deviation of returns is zero, cannot calculate Sharpe ratio for {series_returns.name}")
            return 0 
        sharpe_ratio = np.sqrt(365 / execution_period) * mean_return / std_return
        return sharpe_ratio
    except RuntimeWarning as e:
        raise ValueError(f"RuntimeWarning encountered in get_sharpe_ratio: {e}\n"
                         f"series_returns: {series_returns}\n"
                         f"mean_return: {mean_return}\n"
                         f"std_return: {std_return}")