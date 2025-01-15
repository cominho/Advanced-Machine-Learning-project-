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
from sklearn.linear_model import LinearRegression  
import xgboost as xgb 
import torch 

import backtest as bt 
import adv_machine.utils as ut 
import adv_machine.ml.utils_ml as ut_ml 
import adv_machine.config as cf 
 
import feature_computer as ft
from sklearn.ensemble import RandomForestRegressor





def pearson_cumsom_loss(y_true, y_pred):
    '''
    optmize negative pearson coefficient loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    n = len(y_true)
    y_bar = y_true.mean()
    yhat_bar = y_pred.mean()
    c = 1 / ((y_true - y_bar) ** 2).sum().sqrt()  # constant variable
    b = ((y_pred - yhat_bar) ** 2).sum().sqrt()  # std of pred

    a_i = y_true - y_bar
    d_i = y_pred - yhat_bar
    a = (a_i * d_i).sum()
    gradient = c * (a_i / b - a * d_i / b**3)
    hessian = - (np.matmul(a_i.reshape(-1, 1), d_i.reshape(1, -1)) + np.matmul(d_i.reshape(-1, 1), a_i.reshape(1, -1))) / b ** 3 + \
              3 * a * np.matmul(d_i.reshape(-1, 1), d_i.reshape(1, -1)) / b**5 + a/(n*b**3)
    hessian = hessian - np.ones(shape=(n, n)) * a/b**3
    hessian *= c
    return -gradient, -hessian 
     


def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])


def spearman_loss(ytrue, ypred):
    lenypred = ypred.shape[0]
    lenytrue = ytrue.shape[0]

    ypred_th = torch.tensor(ypred.reshape(1, lenypred), requires_grad=True)
    ytrue_th = torch.tensor(ytrue.reshape(1, lenytrue))

    loss = spearman(ytrue_th, ypred_th, regularization_strength=3)
    print(f'Current loss:{loss}')

    # calculate gradient and convert to numpy
    loss_grads = torch.autograd.grad(loss, ypred_th)[0]
    loss_grads = loss_grads.detach().numpy()

    # return gradient and ones instead of Hessian diagonal
    return loss_grads[0], np.ones(loss_grads.shape)[0]
    

def sharpe_ratio_one_for_all(preds: np.ndarray,dtrain: xgb.DMatrix,temperature: float = 1/20):
    """
    Differentiable "soft top & bottom" Sharpe-ratio objective for XGBoost.
    Grouping is done via dtrain.get_group() => an array of group sizes.

    For each group (one date):
      - Long weights = softmax(preds_group / temperature)
      - Short weights = softmax(-preds_group / temperature)
      => daily L/S return = sum(long_w * actual) - sum(short_w * actual)
    Then Sharpe = mean(daily_returns) / std(daily_returns).
    Minimizing -Sharpe => we push top preds to get large long weight and
                          bottom preds to get large short weight.

    Parameters
    ----------
    preds       : np.ndarray (n_samples,) - current XGBoost predictions
    dtrain      : xgb.DMatrix
                  Must have group sizes set via dtrain.set_group(...)
    temperature : float
                  Smaller => more peaky weighting on extremes. Larger => more spread out.

    Returns
    -------
    grad, hess : np.ndarray of shape (n_samples,) each
    """

    # Convert to torch
    preds_torch = torch.tensor(preds, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(dtrain.get_label(), dtype=torch.float32)

    # Retrieve the group sizes
    groups = dtrain.get_group()  # 1D array, sum(groups) == n_samples
    idx_start = 0

    daily_returns = []

    # We'll slice preds_torch and y_torch by each group block
    for group_size in groups:
        group_size = int(group_size)  # ensure it's int, not np.float
        idx_end = idx_start + group_size
        
        # slice predictions and labels for this group
        preds_g = preds_torch[idx_start:idx_end]
        rets_g  = y_torch[idx_start:idx_end]
        
        idx_start = idx_end

        if group_size < 1:
            # no data in group
            daily_returns.append(torch.tensor(0.0, dtype=torch.float32))
            continue
        
        # 1) "Long" weights = softmax(preds_g / temperature)
        w_long = torch.softmax(preds_g / temperature, dim=0)
        
        # 2) "Short" weights = softmax(-preds_g / temperature)
        w_short = torch.softmax(-preds_g / temperature, dim=0)
        
        # daily L/S return
        day_ret = (w_long * rets_g).sum() - (w_short * rets_g).sum()
        daily_returns.append(day_ret)

    # Now compute overall Sharpe across groups (dates)
    if len(daily_returns) == 0:
        print('Constant gradient')
        return np.zeros_like(preds), np.ones_like(preds)

    else:
        daily_returns_t = torch.stack(daily_returns)
        mean_ret = daily_returns_t.mean()
        std_ret  = daily_returns_t.std() + 1e-9
        sharpe   = mean_ret / std_ret
        loss = -sharpe  # maximize Sharpe => minimize negative Sharpe

    # Backprop
    loss.backward()
    
    # Extract gradient
    grad = preds_torch.grad.detach().cpu().numpy()
    # Dummy Hessian
    hess = np.ones_like(grad, dtype=np.float32)
    return grad, hess 

def sharpe_ratio_multioutput(preds: np.ndarray, dtrain: xgb.DMatrix, temperature: float = 1/20):
    """
    Differentiable "soft top & bottom" Sharpe-ratio objective for multi-output XGBoost.
    Each row of preds represents one date, each column represents one asset.

    For each date (row):
      - Long weights = softmax(preds_row / temperature)
      - Short weights = softmax(-preds_row / temperature)
      => daily L/S return = sum(long_w * actual) - sum(short_w * actual)
    Then Sharpe = mean(daily_returns) / std(daily_returns)

    Parameters
    ----------
    preds       : np.ndarray (n_dates, n_assets) - current XGBoost predictions
    dtrain      : xgb.DMatrix
                  Labels should have same shape as preds (n_dates, n_assets)
    temperature : float
                  Smaller => more peaky weighting on extremes. Larger => more spread out.

    Returns
    -------
    grad, hess : np.ndarray of shape (n_dates * n_assets,) each
    """
    # Get the shape directly from the predictions
    n_dates, n_assets = preds.shape
    
    # Convert to torch
    preds_torch = torch.tensor(preds, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(dtrain.get_label().reshape(-1, n_assets), dtype=torch.float32)

    daily_returns = []

    # Process each date (row)
    for preds_row, rets_row in zip(preds_torch, y_torch):
        # 1) "Long" weights = softmax(preds_row / temperature)
        w_long = torch.softmax(preds_row / temperature, dim=0)/2
        
        # 2) "Short" weights = softmax(-preds_row / temperature)
        w_short = torch.softmax(-preds_row / temperature, dim=0)/2
        
        # daily L/S return
        day_ret = (w_long * rets_row).sum() - (w_short * rets_row).sum()
        daily_returns.append(day_ret)

    # Compute overall Sharpe across dates
    daily_returns_t = torch.stack(daily_returns)
    mean_ret = daily_returns_t.mean()
    std_ret = daily_returns_t.std() + 1e-9
    sharpe = mean_ret / std_ret
    loss = -sharpe  # maximize Sharpe => minimize negative Sharpe

    # Backprop
    loss.backward()
    
    # Extract gradient and reshape to match XGBoost's expected shape (n_samples, n_targets)
    grad = preds_torch.grad.detach().cpu().numpy()  # Already in shape (n_dates, n_assets)
    # Hessian with same shape as gradient
    hess = np.ones_like(grad, dtype=np.float32)
    
    return grad, hess 


def sharpe_ratio_correlation_multioutput(preds: np.ndarray, dtrain: xgb.DMatrix):
    """
    Enhanced Sharpe ratio objective using group-based correlations from DMatrix
    
    Parameters
    ----------
    preds   : np.ndarray - predictions
    dtrain  : xgb.DMatrix - must have groups set via set_group()
    """
    # Convert to torch tensors
    preds_torch = torch.tensor(preds, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(dtrain.get_label(), dtype=torch.float32)
    
    # Get groups from DMatrix
    groups = dtrain.get_group()
    era_correlations = []
    
    # Track position in arrays
    start_idx = 0
    
    # Calculate correlation for each group/era
    for group_size in groups:
        end_idx = start_idx + int(group_size)
        
        # Get predictions and labels for this group
        preds_group = preds_torch[start_idx:end_idx]
        y_group = y_torch[start_idx:end_idx]
        
        if len(preds_group) > 1:  # Need at least 2 points for correlation
            # Normalize within group
            preds_n = preds_group - preds_group.mean()
            preds_n = preds_n / (preds_n.norm() + 1e-8)
            
            y_n = y_group - y_group.mean()
            y_n = y_n / (y_n.norm() + 1e-8)
            
            # Calculate group correlation
            group_corr = (preds_n * y_n).sum()
            era_correlations.append(group_corr)
        
        start_idx = end_idx
    
    if not era_correlations:
        return np.zeros_like(preds), np.ones_like(preds)
    
    # Stack correlations and calculate Sharpe
    correlations = torch.stack(era_correlations)
    mean_corr = correlations.mean()
    std_corr = correlations.std() + 1e-8
    sharpe = mean_corr / std_corr
    
    loss = -sharpe  # Minimize negative Sharpe
    loss.backward()
    
    grad = preds_torch.grad.detach().numpy()
    hess = np.ones_like(grad)
    
    return grad, hess 


def spearman_sharpe_ratio_spearman_correlation_multioutput(preds: np.ndarray, dtrain: xgb.DMatrix, regularization_strength: float = 1.0):
    """
    Enhanced Sharpe ratio objective using day-by-day Spearman rank correlations
    using differentiable soft ranking
    
    Parameters
    ----------
    preds   : np.ndarray - predictions
    dtrain  : xgb.DMatrix - must have groups set via set_group()
    regularization_strength: float - controls how close soft ranks are to true ranks (smaller = closer)
    """
    # Convert to torch tensors
    preds_torch = torch.tensor(preds, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(dtrain.get_label(), dtype=torch.float32)
    
    era_correlations = []
    
    # Calculate Spearman correlation for each day
    for preds_day, y_day in zip(preds_torch, y_torch):
        if len(preds_day) > 1:  # Need at least 2 points for correlation
            # Use soft ranking with L2 regularization
            preds_ranks = soft_rank(
                preds_day.reshape(1, -1),
                regularization="l2",
                regularization_strength=regularization_strength
            ).squeeze()
            
            y_ranks = soft_rank(
                y_day.reshape(1, -1),
                regularization="l2",
                regularization_strength=regularization_strength
            ).squeeze()
            
            # Normalize ranks
            preds_ranks = (preds_ranks - preds_ranks.mean()) / (preds_ranks.std() + 1e-8)
            y_ranks = (y_ranks - y_ranks.mean()) / (y_ranks.std() + 1e-8)
            
            # Calculate day Spearman correlation
            day_corr = (preds_ranks * y_ranks).sum()
            era_correlations.append(day_corr)
    
    if not era_correlations:
        return np.zeros_like(preds), np.ones_like(preds)
    
    # Stack correlations and calculate Sharpe
    correlations = torch.stack(era_correlations)
    mean_corr = correlations.mean()
    std_corr = correlations.std() + 1e-8
    sharpe = mean_corr / std_corr
    
    loss = -sharpe  # Minimize negative Sharpe
    loss.backward()
    
    grad = preds_torch.grad.detach().numpy()
    hess = np.ones_like(grad)
    
    return grad, hess 

def spearman_correlation_multioutput(preds: np.ndarray, dtrain: xgb.DMatrix, regularization_strength: float = 1.0):
    """
    Objective using day-by-day Spearman rank correlations
    using differentiable soft ranking
    
    Parameters
    ----------
    preds   : np.ndarray - predictions
    dtrain  : xgb.DMatrix - must have groups set via set_group()
    regularization_strength: float - controls how close soft ranks are to true ranks (smaller = closer)
    """
    # Convert to torch tensors
    preds_torch = torch.tensor(preds, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(dtrain.get_label(), dtype=torch.float32)
    
    era_correlations = []
    
    # Calculate Spearman correlation for each day
    for preds_day, y_day in zip(preds_torch, y_torch):
        if len(preds_day) > 1:  # Need at least 2 points for correlation
            # Use soft ranking with L2 regularization
            preds_ranks = soft_rank(
                preds_day.reshape(1, -1),
                regularization="l2",
                regularization_strength=regularization_strength
            ).squeeze()
            
            y_ranks = soft_rank(
                y_day.reshape(1, -1),
                regularization="l2",
                regularization_strength=regularization_strength
            ).squeeze()
            
            # Normalize ranks
            preds_ranks = (preds_ranks - preds_ranks.mean()) / (preds_ranks.std() + 1e-8)
            y_ranks = (y_ranks - y_ranks.mean()) / (y_ranks.std() + 1e-8)
            
            # Calculate day Spearman correlation
            day_corr = (preds_ranks * y_ranks).sum()
            era_correlations.append(day_corr)
    
    if not era_correlations:
        return np.zeros_like(preds), np.ones_like(preds)
    
    # Simply take the mean correlation across all eras
    mean_corr = torch.stack(era_correlations).mean()
    
    loss = -mean_corr  # Minimize negative correlation
    loss.backward()
    
    grad = preds_torch.grad.detach().numpy()
    hess = np.ones_like(grad)
    
    return grad, hess 









