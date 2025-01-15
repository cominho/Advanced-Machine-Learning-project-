from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn import model_selection, metrics 
import pandas as pd
import numpy as np
from scipy.stats import spearmanr 
import copy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression 
from sklearn.model_selection import TimeSeriesSplit  
from sklearn.base import BaseEstimator, RegressorMixin 
import xgboost as xgb 
import itertools 
import json   

import adv_machine.backtest.backtester as bt 
import adv_machine.utils as ut 
import adv_machine.universe as un  
import adv_machine.config as cf 
import adv_machine.feature_computer as ft 
import adv_machine.get_prices as gp
import adv_machine.backtest.metrics as metrics 
from adv_machine.log import _print , _print_error  
from adv_machine.parallel.parallel_chunk import parallelize_chunk  

from functools import partial  
import json 
import os 


 

def compute_prices_features_target(args):
    """Compute features for a single product"""
    product, config_features, config_target, verbose = args
   
    
    product_configs_main = []
    
    
    for config in config_features + [config_target]:
        product_config = copy.deepcopy(config)
        product_config[list(product_config.keys())[0]]['collect']['product'] = product
        product_configs_main.append(product_config)

    df_features, label_to_ohlc = ft.feature_computer(product_configs_main, return_ohlc=True, verbose=0)

    df_target = df_features[ut.get_name_config(config_target)]
    df_target = df_target.shift(-1)
    df_features = df_features.drop(columns=[ut.get_name_config(config_target)], axis=1)
   


    df_prices = label_to_ohlc.get(product, gp.get_stock_ohlc(product))
    
    
    date_to_ohlc = df_prices.to_dict(orient='index')
    date_to_prices = {}
    for date, ohlc_data in date_to_ohlc.items():
        date_to_prices.setdefault(date, {})[product] = ohlc_data
    
    return product, df_features, df_target, date_to_prices 
def compute_prices_features_target_chunk(chunk_args):
    """Process a chunk of products for price and feature computation"""
    chunk_results = []
    for args in chunk_args:
        product, config_features, config_target, verbose = args
        try:
            result = compute_prices_features_target(args)
            if result[1] is not None:  # Check if features were computed
                chunk_results.append(result)
        except Exception as e:
            _print_error(f"Error processing product {product}: {str(e)}")
            continue
    return chunk_results 





 

def evaluate_params(args):
    model,params, params_idx, product_to_df_features, product_to_target,period_train,period_test, config_aggregation, fees_bps, date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose = args
    
    
        
    model = model(**params)
    first_date = list(date_to_product_universe.keys())[0]
    _print(f"Starting from first date: {first_date}", 1, verbose)
    
    # Create combined DataFrames for features and targets separately
    features_df = pd.concat(product_to_df_features.values(), keys=product_to_df_features.keys(), axis=1)
    features_df = ut.sort_date_index(features_df)
    features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
    dates_features = features_df.index

    targets_df = pd.concat(product_to_target.values(), keys=product_to_target.keys(), axis=1)
    targets_df.columns = [f"{product}_target" for product in product_to_target.keys()]
    targets_df = ut.sort_date_index(targets_df)
    dates_target = targets_df.index

    dates_features_target = list(set(dates_features).intersection(dates_target))
    
    # Combine features and targets
    combined_df = pd.concat([features_df, targets_df], axis=1)
    combined_df = combined_df.loc[dates_features_target]
    combined_df = ut.sort_date_index(combined_df)
    combined_df = combined_df.loc[combined_df.index >= first_date]
    dates = combined_df.index
    
    # Split into X and y
    feature_cols = [col for col in combined_df.columns if not col.endswith('_target')]
    target_cols = [col for col in combined_df.columns if col.endswith('_target')]
    
    X = combined_df[feature_cols]
    y = combined_df[target_cols]
    _print(f"Final X shape: {X.shape}, y shape: {y.shape}", 1, verbose)

    date_to_product_signal = {}
    date_to_product_score = {}
    report = {}

    smooth = params.get('smooth',0)
    only_available_products = params.get('only_available_products',False)
    params = {k: v for k, v in params.items() if k not in ['smooth','only_available_products']}

     
    # Walk-forward approach
    for i in range(0, len(dates), period_test):
        _print(f"Starting walk-forward iteration {i//period_test + 1}", 1, verbose)
        report[i] = {'product':[],'date_start_train':None,'date_end_train':None,'date_start_test':None,'date_end_test':None}
        
        # Calculate training dates
        train_start = i
        train_end = min(i + period_train, len(dates))
        train_dates = dates[train_start:train_end]
        tolerance_rate = 0.05 

        _print(f"Training period: {train_dates[0]} to {train_dates[-1]}", 1, verbose)
        
        # Calculate prediction dates
        test_start = train_end
        test_end = min(test_start + period_test, len(dates))
        test_dates = dates[test_start:test_end]
        
        if len(test_dates) < 1:
            _print('Not enough predict dates. We stop the validation', 1, verbose)
            break 
        _print(f"Testing period: {test_dates[0]} to {test_dates[-1]}", 1, verbose)
        
        # Create dictionaries for train/test data before the loop
        
        product_to_df_features_train = {}
        product_to_df_features_test = {}
        product_to_target_train = {}
        product_to_target_test = {}
        products_processed = 0
        
        report[i]['date_start_train'] = train_dates[0]
        report[i]['date_end_train'] = train_dates[-1]
        
        report[i]['date_start_test'] = test_dates[0]
        report[i]['date_end_test'] = test_dates[-1]
        available_products = set()
        for date in test_dates:
            if ut.format_datetime(date) in date_to_product_universe.keys() : 
                available_products.update(date_to_product_universe[ut.format_datetime(date)])
        available_products = list(available_products)
        for product, df_feature in product_to_df_features.items():
            # Skip products not in target data
            if product not in product_to_target:
                continue 
            if only_available_products :
                if product not in available_products : 
                    continue 
  
            tolerance_size = len(train_dates)*(1-tolerance_rate)
            train_dates_product = list(set(df_feature.index).intersection(set(train_dates)))
            if len(train_dates_product) < tolerance_size:
                continue 
            missing_size  = len(train_dates) - len(train_dates_product)
            _print(f"Missing {missing_size} dates for product {product}", 1, verbose)
            try:
                # Get features and targets for the current period
                features_train = df_feature.loc[train_dates_product]
                features_test = df_feature.loc[test_dates]
                target_train = product_to_target[product].loc[train_dates_product]
                target_test = product_to_target[product].loc[test_dates]

                # Only include if we have data for both train and test periods
                if (not features_train.empty and not features_test.empty and 
                    not target_train.empty and not target_test.empty):
                    product_to_df_features_train[product] = features_train
                    product_to_df_features_test[product] = features_test
                    product_to_target_train[product] = target_train
                    product_to_target_test[product] = target_test
                    report[i]['product'].append(product)
                    

                    products_processed += 1

            except Exception as e:
                _print(f"Error processing product {product}: {str(e)}", 1, verbose)

                continue 
        report[i]['products_processed'] = products_processed

        # Skip iteration if no valid data
        if not product_to_df_features_train or not product_to_target_train:
            _print(f"No valid data for period {i}, skipping...", 1, verbose)
            continue

        model.fit(product_to_df_features_train, product_to_target_train)

        date_to_product_score_chunk = model.predict(product_to_df_features_test)
        num_predictions = sum(len(scores) for scores in date_to_product_score_chunk.values())

        
        for date, product_to_score in date_to_product_score_chunk.items():
            date_to_product_score[date] = product_to_score
              

        # Convert report dates to string format for JSON serialization
        try : 
            json_report = {}
            for period, period_data in report.items():
                json_report[ut.make_serializable(period)] = {
                    'product': ut.make_serializable(period_data['product']),
                    'date_start_train': ut.make_serializable(period_data['date_start_train']),
                    'date_end_train': ut.make_serializable(period_data['date_end_train']),
                    'date_start_test': ut.make_serializable(period_data['date_start_test']),
                    'date_end_test': ut.make_serializable(period_data['date_end_test']),
                    'products_processed': ut.make_serializable(period_data.get('products_processed', 0))
                }
            os.makedirs('output', exist_ok=True)
            with open(f'output/report_validationPipeline_{params_idx}.json', 'w') as f:
                json.dump(json_report, f, indent=4)
        except Exception as e: 
            _print_error(f'There is an issue with json : {str(e)} ')
        # Create output directory if it doesn't exist
        
    
    # After the walk-forward loop, ensure we have predictions before proceeding
    if not date_to_product_score:
        _print("No valid predictions generated", 1, verbose)
        return {'params': params, 'score': '-inf'}

    method_aggregation = 'bottom_top'
    signal_type = 'allocation_fraction'
    if smooth > 0:
        _print(f"Applying exponential smoothing with span={smooth}", 1, verbose)
        df_score = pd.DataFrame(list(date_to_product_score.values()),index=list(date_to_product_score.keys()))
        _print(f"df_score smooth : {df_score.head(5)}", 1, verbose)
        for column in df_score.columns:
            # Store original NaN positions
            nan_mask = df_score[column].isna()
            # Apply ewm only on non-NaN values
            df_score[column] = df_score[column].ewm(span=smooth).mean()
            # Restore NaN values to their original positions
            df_score.loc[nan_mask, column] = np.nan
        date_to_product_score = df_score.to_dict(orient='index')

    date_to_product_score_universe = {}
    
    for date, product_score in date_to_product_score.items():
        date_without_time = ut.format_datetime(date)
        available_products = date_to_product_universe.get(date_without_time,[])
        if available_products:
            product_score_available = {product: score for product, score in product_score.items() if product in available_products}
            if product_score_available : 
                date_to_product_score_universe[date] = product_score_available 
    
    
    date_to_product_signal = ft.score_to_aggregation(
        date_to_product_score=date_to_product_score_universe,
        method_aggregation=method_aggregation,
        config_aggregation=config_aggregation,
        product_to_last_date=product_to_last_date      
    ) 
    json_compatible_signals = {}
    json_compatible_scores = {}

    for date, product_scores in date_to_product_score_universe.items():
        json_compatible_scores[date.strftime('%Y-%m-%d %H:%M:%S')] = ut.make_serializable(product_scores)

    for date, product_signals in date_to_product_signal.items():
        json_compatible_signals[date.strftime('%Y-%m-%d %H:%M:%S')] = ut.make_serializable(product_signals)
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save signals
    signals_path = os.path.join(output_dir, f'aggregation_signals_{params_idx}.json')
    with open(signals_path, 'w') as f:
        json.dump(json_compatible_signals, f, indent=4)

    # Save scores
    scores_path = os.path.join(output_dir, f'aggregation_scores_{params_idx}.json')
    with open(scores_path, 'w') as f:
        json.dump(json_compatible_scores, f, indent=4)
        
    
    
    # Save date_to_product_signal to JSON
    
    # Create output directory if it doesn't exist
   
    
    # Convert datetime index to string format for JSON serialization
    
    
    
    
    series_cumulative_pnl = bt.calculate_cumulative_pnl_v2(
        active_universe = products,
        date_to_products_signal=date_to_product_signal,
        date_to_product_prices=date_to_product_prices,
        fees_bps=fees_bps,
        signal_type = signal_type,
        notional=10000,
        verbose=0
    ) 
    
    series_return = ut.compute_returns_from_cumulative_pnl(series_cumulative_pnl, initial_capital=10000)
    score = metrics.get_sharpe_ratio(series_return)
    score = np.round(score, 2)

    # Helper function to make values JSON serializable
    

    # Convert result dictionary with nested serialization
    result = {
        'params': params ,
        'score': score,
        'period_train': period_train,
        'period_test': period_test 
    }
    _print(f'Score for param : {result}', 1, verbose)
    
   
    
    return {
        'params': params,
        'score': score
    }
def evaluate_params_chunk(chunk_args):
    """Process a chunk of parameter combinations"""
    
    chunk_results = []
    for args in chunk_args:
        model, params, param_idx, product_to_df_features, product_to_target,period_train,period_test, config_aggregation, fees_bps, date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose = args
        try:
            result = evaluate_params(args)
            if result is not None:
                chunk_results.append(result)
        except Exception as e:
            _print_error(f"Error processing params {params}: {str(e)}")
            continue
    return chunk_results 
def is_fitted(model):
    """
    Check if an XGBoost model has been fitted.
    
    Args:
        model: XGBoost model instance to check
        
    Returns:
        bool: True if the model has been fitted, False otherwise
    """
    try:
        print(model.get_booster())
        return (model.get_booster() is not None)
    except:
        return False
def keep_multiindex_prediction(df_multi_index, model):
    """
    Make predictions while preserving multi-index information.
    
    Parameters:
    -----------
    df_multi_index : pd.DataFrame
        DataFrame with multi-index (date, product)
    model : object
        Fitted model with predict method (e.g., XGBoost)
        
    Returns:
    --------
    pd.Series
        Predictions with the same multi-index as input DataFrame
    """
    # Store original index
    original_index = df_multi_index.index
    
    # Reset index for prediction
    X = df_multi_index.reset_index(drop=True)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create Series with original multi-index
    return pd.Series(predictions, index=original_index)
def neutralize(X, series_prediction, proportion=1.0):
    """
    Neutralizes predictions with respect to given exposures (features).
    
    Args:
        X (pd.DataFrame/np.ndarray): Feature matrix of exposures to neutralize against
        series_prediction (pd.Series): Predictions to be neutralized
        proportion (float): Proportion of exposure to neutralize (0 to 1)
    
    Returns:
        pd.Series: Neutralized and standardized predictions
        
    Raises:
        ValueError: If inputs have incompatible shapes or invalid values
    """
    if not 0 <= proportion <= 1:
        raise ValueError("proportion must be between 0 and 1")
        
    if len(X) != len(series_prediction):
        raise ValueError("X and series_prediction must have the same length")
    
    scores = series_prediction.copy()
    exposures = X.values if hasattr(X, 'values') else np.array(X)
    
    # Add constant column for complete neutralization
    exposures = np.hstack((
        exposures,
        np.full((len(exposures), 1), np.mean(scores))
    ))
    
    # Compute neutralization using pseudo-inverse
    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    
    # Standardize the results
    return scores / (scores.std() or 1.0)  # Avoid division by zero

