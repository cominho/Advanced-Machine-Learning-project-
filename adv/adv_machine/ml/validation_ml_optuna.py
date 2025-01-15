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
import json 
import os 
import optuna 
import logging 
import sys    

import adv_machine.backtest.backtester as bt 
import adv_machine.utils as ut 
import adv_machine.config as cf 
import adv_machine.feature_computer as ft 
import adv_machine.universe as un 
import adv_machine.ml.custom_loss as cl 
import adv_machine.get_prices as gp 
import adv_machine.backtest.metrics as metrics 
from functools import partial 
from adv_machine.parallel.parallel_chunk import parallelize_chunk 
from adv_machine.log import _print 
import adv_machine.ml.utils_ml as ut_ml 

def objective_xgb(trial, fixed_args):
    """
    Creates an objective function for hyperopt optimization.
    
    Args:
        params: Dictionary of parameters to optimize (provided by hyperopt)
        fixed_args: Tuple containing all other required arguments for evaluate_params:
            (model, product_to_df_features, product_to_target, period_train, 
             period_test, config_aggregation, fees_bps, date_to_product_prices)
    
    Returns:
        dict: Contains loss (negative score for maximization), status, and parameters
    """
    # Unpack the fixed arguments
    (model, product_to_df_features, product_to_target, period_train, 
     period_test, config_aggregation, fees_bps, date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose) = fixed_args 
    trial_number = trial.number
    params = {'n_estimators':10000,
              'verbosity':0,
              'n_jobs':-1,
              'objective':'reg:squarederror',
              'smooth': trial.suggest_int('smooth',5,15,step=5),
              'feature_neutralization':trial.suggest_float('feature_neutralization',0,0.8,step=0.05),
              'learning_rate': trial.suggest_float('learning_rate', 1e-9, 1e-3, log=True),
              'max_depth':trial.suggest_int('max_depth', 2, 7),
              'col_sample_bytree':trial.suggest_float('col_sample_bytree', 0.1, 1, step=0.05),
              'subsample':trial.suggest_float('subsample', 0.1, 1, step=0.05),
              'convert_target':True,
                      }
    
    # Create args tuple for evaluate_params
    args = (model, params,trial_number, product_to_df_features, product_to_target, 
            period_train, period_test, config_aggregation, fees_bps, 
            date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose)
    
    # Get evaluation result
    result = ut_ml.evaluate_params(args)
    
    # Return in hyperopt format (negative score because hyperopt minimizes)
    return result['score']
def objective_xgb_multioutput_loss(trial, fixed_args):
    """
    Creates an objective function for hyperopt optimization.
    
    Args:
        params: Dictionary of parameters to optimize (provided by hyperopt)
        fixed_args: Tuple containing all other required arguments for evaluate_params:
            (model, product_to_df_features, product_to_target, period_train, 
             period_test, config_aggregation, fees_bps, date_to_product_prices)
    
    Returns:
        dict: Contains loss (negative score for maximization), status, and parameters
    """
    # Unpack the fixed arguments
    (model, product_to_df_features, product_to_target, period_train, 
     period_test, config_aggregation, fees_bps, date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose) = fixed_args 
    trial_number = trial.number
    params = {'n_estimators':10000,
              'verbosity':0,
              'n_jobs':-1,
              'smooth': trial.suggest_int('smooth',5,15,step=5),
              'learning_rate': trial.suggest_float('learning_rate', 1e-15, 1e-8, log=True),
              'max_depth':trial.suggest_int('max_depth', 2, 7),
              'col_sample_bytree':trial.suggest_float('col_sample_bytree', 0.01, 1, step=0.01),
              'reg_lambda':trial.suggest_float('reg_lambda', 1e-15, 1, log=True),
              'grow_policy': trial.suggest_categorical('grow_policy',['depthwise','lossguide']),
              'min_split_loss':trial.suggest_float('min_split_loss', 1e-6, 1e1, log=True),
              'subsample': 1,
              'convert_target':True,
              'multi_strategy': trial.suggest_categorical('multi_strategy',['one_output_per_tree','multi_output_tree']),
                      }
    temperature = trial.suggest_float('temperature', 1e-3, 1e1, log = True)
    obj = partial(cl.sharpe_ratio_multioutput, temperature=temperature)
    params['obj'] = obj 
    params['only_available_products'] = True 
    # Create args tuple for evaluate_params
    args = (model, params,trial_number, product_to_df_features, product_to_target, 
            period_train, period_test, config_aggregation, fees_bps, 
            date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose)
    
    # Get evaluation result
    result = ut_ml.evaluate_params(args)
    
    # Return in hyperopt format (negative score because hyperopt minimizes)
    return result['score']

def objective_xgb_rmse(trial, fixed_args):
    """
    Creates an objective function for hyperopt optimization.
    
    Args:
        params: Dictionary of parameters to optimize (provided by hyperopt)
        fixed_args: Tuple containing all other required arguments for evaluate_params:
            (model, product_to_df_features, product_to_target, period_train, 
             period_test, config_aggregation, fees_bps, date_to_product_prices)
    
    Returns:
        dict: Contains loss (negative score for maximization), status, and parameters
    """
    # Unpack the fixed arguments
    (model, product_to_df_features, product_to_target, period_train, 
     period_test, config_aggregation, fees_bps, date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose) = fixed_args 
    trial_number = trial.number
    params = {'n_estimators':10000,
              'verbosity':0,
              'n_jobs':-1,
              'smooth': trial.suggest_int('smooth',5,15,step=5),
              'learning_rate': trial.suggest_float('learning_rate', 1e-15, 1e-5, log=True),
              'max_depth':trial.suggest_int('max_depth', 2, 7),
              'col_sample_bytree':trial.suggest_float('col_sample_bytree', 0.01, 1, step=0.01),
              'subsample': trial.suggest_float('subsample', 0.01, 1, step=0.01),
              'convert_target':True,
              'multi_strategy': 'one_output_per_tree',
                      }
    params['only_available_products'] = True 
    # Create args tuple for evaluate_params
    args = (model, params,trial_number, product_to_df_features, product_to_target, 
            period_train, period_test, config_aggregation, fees_bps, 
            date_to_product_prices,date_to_product_universe,product_to_last_date,products,verbose)
    
    # Get evaluation result
    result = ut_ml.evaluate_params(args)
    
    # Return in hyperopt format (negative score because hyperopt minimizes)
    return result['score']

def grid_search_optuna(objective,study_name,model, end_date_training,baseline_universe,config_universe, 
                        config_target, config_features, config_aggregation, fees_bps,period_train, period_test, n_jobs=0, max_evals=25,load_if_exists=False, verbose=0):
    """
    Hyperopt-based parameter optimization function.
    Uses the same interface as grid_search_backtest but performs Bayesian optimization.
    """
    universe = un.Universe(baseline_universe, config_universe)
    universe.compute_universe(end_date_training)
    date_to_product_universe = universe.date_to_product_universe 
    products = universe.active_universe 
    _print(f"Computing universe for end date: {end_date_training}", 1, verbose)
    _print(f"Number of products in active universe: {len(products)}", 1, verbose)
    _print(f"Products: {products}", 2, verbose)  # More detailed logging at verbose level 2
    # Generate all parameter combinations
    
    
   
    
    
    # Computing prices
    _print('---------- Computing Prices, Features and Target -----------', 1, verbose)
    args_compute_prices_features_target = [(product,config_features, config_target, verbose) for product in products]
    if n_jobs == -1:
        prices_features_target_results = parallelize_chunk(
            ut_ml.compute_prices_features_target_chunk,
            args_compute_prices_features_target,
            desc="Computing prices, features and target"
        )
    else:
        prices_features_target_results = []
        for args in tqdm(args_compute_prices_features_target,total = len(args_compute_prices_features_target),desc='Computing prices, features and target'):
            prices_features_target_results.append(ut_ml.compute_prices_features_target(args))
    date_to_product_prices = {}
    product_to_df_features = {}
    product_to_target = {}
    product_to_last_date = {}
    
    _print("Processing results for each product...", 1, verbose)
    for product, df_features, df_target, date_to_prices in prices_features_target_results:
        _print(f"Processing data for product: {product}", 2, verbose)
        
        _print(f"Adding price data for {len(date_to_prices)} dates", 2, verbose)
        for date, prices in date_to_prices.items():
                if date in date_to_product_prices:
                    date_to_product_prices[date].update(prices)
                else:
                    date_to_product_prices[date] = prices
        
        _print(f"Adding features (shape: {df_features.shape}) and target (shape: {df_target.shape})", 2, verbose)
        product_to_df_features[product] = df_features
        product_to_target[product] = df_target
        product_to_last_date[product] = ut.format_datetime(df_features.index[-1])
    _print(f"Completed processing for {len(product_to_df_features)} products", 1, verbose)
    
    # Create fixed arguments tuple for evaluate_params_hyperopt
    fixed_args = (
        model,
        product_to_df_features,
        product_to_target,
        period_train,
        period_test,
        config_aggregation,
        fees_bps,
        date_to_product_prices,
        date_to_product_universe,
        product_to_last_date,
        products,
        verbose
    )
    
    # Create the objective function
    objective = partial(objective, fixed_args=fixed_args)
    
    # Unique identifier of the study.
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction="maximize", storage=storage_name,study_name=study_name,load_if_exists=load_if_exists,sampler = optuna.samplers.TPESampler())
    
    study.optimize(func=objective, n_trials=max_evals,n_jobs= 1,show_progress_bar = True)

    # Create list of dictionaries with parameters and scores
    try : 
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_dict = {
                    'params': ut.make_serializable(trial.params),
                    'score': ut.make_serializable(trial.value)
                }
                trials_data.append(trial_dict)

        # Save to JSON file
        json_filename = f"{study_name}_trials.json"
        with open(json_filename, 'w') as f:
            json.dump(trials_data, f, indent=4)
    except Exception as e:
        _print(f"Error saving trials data to JSON: {e}", 1, verbose)

    print(study.best_trial)

    return study 

def report_optuna(study_name):
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    # Original visualizations
    fig_objective = optuna.visualization.plot_optimization_history(study)
    fig_objective.show()

    fig_params = optuna.visualization.plot_param_importances(study)
    fig_params.show()

    fig_slice = optuna.visualization.plot_slice(study)
    fig_slice.show()




    if study.best_trial is not None:
        print(f'Best params: {study.best_params}')
        print(f'Best value: {study.best_value}')
    else:
        print("No completed trials found.")

    # Filter out trials that don't have a value
    trials_with_values = [t for t in study.trials if t.value is not None]

    if not trials_with_values:
        print("No trials with valid values to report.")
        return

    # Sort the trials based on their value
    top_trials = sorted(trials_with_values, key=lambda t: t.value, reverse=True)[:10]

    print('\nTop 10 Trials:')
    for i, trial in enumerate(top_trials, 1):
        print(f'\nRank {i}:')
        print(f'  Value: {trial.value}')
        print(f'  Params: {trial.params}')
    
    # Initialize trials object to store results
    