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
from functools import partial  

import backtest as bt 
import adv_machine.utils as ut 
import adv_machine.ml.utils_ml as ut_ml 
import adv_machine.ml.custom_loss as cl  
import adv_machine.config as cf 
import feature_computer as ft
from sklearn.ensemble import RandomForestRegressor













class PLS_Regressor_average_feature(BaseEstimator):
    def __init__(self, n_components=2, method_scale='z_score',convert_target = None,  **pls_params):
        self.n_components = n_components
        self.method_scale = method_scale
        self.pls_params = pls_params
        self.model = PLSRegression(n_components=n_components)
        self.train_mean_ = None
        self.train_std_ = None
        self.feature_names_ = None
        self.target_names_ = None
        self.convert_target = convert_target 
        
    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")
            
        # Validate that all feature DataFrames have the same columns
        feature_columns = None
        for product, df_feature in product_to_df_feature.items():
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg)

        # Create features DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(), axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        
        # Get unique feature names (without product prefix)
        feature_names = set('_'.join(col.split('_')[1:]) for col in features_df.columns)
        
        # Calculate mean for each feature across products more efficiently
        feature_groups = {}
        for col in features_df.columns:
            feature = '_'.join(col.split('_')[1:])  # Get feature name without product prefix
            if feature not in feature_groups:
                feature_groups[feature] = []
            feature_groups[feature].append(col)
        
        # Create averaged features DataFrame in one go
        averaged_features = pd.DataFrame(
            {feature: features_df[cols].mean(axis=1) 
             for feature, cols in feature_groups.items()},
            index=features_df.index
        )
        
        # Create targets DataFrame
        targets_df = pd.concat(product_to_target.values(), keys=product_to_target.keys(), axis=1)
        targets_df.columns = [f"{product}_target" for product in product_to_df_feature.keys()]
        
        # Apply z-score normalization to targets if specified
        if self.convert_target :
            # Z-score normalize each row
            row_means = targets_df.mean(axis=1)
            row_stds = targets_df.std(axis=1)
            targets_df = targets_df.sub(row_means, axis=0).div(row_stds, axis=0)
            
            
        
        # Combine averaged features with targets
        combined_df = pd.concat([averaged_features, targets_df], axis=1)
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.dropna()
        
        # Add the qid column with a unique value for each unique date
        
        return combined_df
    
    def _scale_features(self, X, fit=False):
        if self.method_scale == 'z_score':
            if fit:
                self.train_mean_ = X.mean()
                self.train_std_ = X.std()
                # Handle zero/small standard deviations
                self.train_std_ = self.train_std_.replace(0, 1)
                self.train_std_[self.train_std_ < 1e-8] = 1
            
            X_scaled = X.copy()
            for column in X.columns:
                X_scaled[column] = (X[column] - self.train_mean_[column]) / self.train_std_[column]
            
        elif self.method_scale == 'pct_change':
            X_scaled = X.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method_scale}")
            
        return X_scaled
    
    def fit(self, product_to_df_feature, product_to_target):
        """
        Fit the model using product features and targets
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
        product_to_target : dict
            Dictionary mapping products to their target DataFrames
        """
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)
        
        # Split features and targets
        self.feature_names_ = [col for col in combined_df.columns if not col.endswith('_target')]
        self.target_names_ = [col for col in combined_df.columns if col.endswith('_target')]
        
        X = combined_df[self.feature_names_]
        y = combined_df[self.target_names_]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, product_to_df_feature):
        """
        Predict using the fitted model
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
            
        Returns:
        --------
        date_to_product_score : dict
            Dictionary with dates as keys and nested dictionaries as values,
            where nested dictionaries map products to their predictions
        """
        # Create features DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(), axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        
        # Get unique feature names (without product prefix)
        feature_names = set('_'.join(col.split('_')[1:]) for col in features_df.columns)
        
        # Calculate mean for each feature across products more efficiently
        feature_groups = {}
        for col in features_df.columns:
            feature = '_'.join(col.split('_')[1:])  # Get feature name without product prefix
            if feature not in feature_groups:
                feature_groups[feature] = []
            feature_groups[feature].append(col)
        
        # Create averaged features DataFrame in one go
        averaged_features = pd.DataFrame(
            {feature: features_df[cols].mean(axis=1) 
             for feature, cols in feature_groups.items()},
            index=features_df.index
        )
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names_) - set(averaged_features.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = averaged_features[self.feature_names_]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=False)
        
        # Make predictions
        y_pred = pd.DataFrame(
            self.model.predict(X_scaled),
            index=X.index,
            columns=self.target_names_
        )
        
        # Apply z-score normalization if specified
        if self.convert_target : 
            # Z-score normalize each row
            row_means = y_pred.mean(axis=1)
            row_stds = y_pred.std(axis=1)
            y_pred = y_pred.sub(row_means, axis=0).div(row_stds, axis=0)
            
            
        
        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        for date in y_pred.index:
            product_scores = {}
            # First pass: apply tanh if needed
            for col in y_pred.columns:
                product = col.replace('_target', '')
                score = y_pred.loc[date, col]
                product_scores[product] = score 
            date_to_product_score[date] = product_scores
            
            
            
        return date_to_product_score 
class LinearRegression_one_fit_all(BaseEstimator):
    def __init__(self, method_scale='z_score',convert_target = None,  **regression_params):
        self.method_scale = method_scale
        self.regression_params = regression_params
        self.model = LinearRegression(**regression_params)
        self.train_mean_ = None
        self.train_std_ = None
        self.feature_names_ = None
        self.target_names_ = None
        self.convert_target = convert_target 
        
    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")
        
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            X = df_feature.dropna().copy(deep=True)
            y = product_to_target[product]
            
            # Handle y based on its type
            if isinstance(y, pd.Series):
                y.name = 'target'
            elif isinstance(y, pd.DataFrame):
                if len(y.columns) > 1:
                    raise ValueError(f"Target DataFrame for {product} has more than one column")
                y = y.rename(columns={y.columns[0]: 'target'})
            else:
                raise ValueError(f"Unsupported target type for {product}: {type(y)}")
            
            df = pd.concat([X, y], axis=1)
            df = ut.sort_date_index(df)
            
            df.index = pd.MultiIndex.from_product([[product], df.index], names=['product', 'date'])
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.swaplevel().sort_index()
        combined_df = combined_df.dropna()
        
        # Store feature names and target name
        self.feature_names_ = [col for col in combined_df.columns if col != 'target']
        self.target_names_ = [col for col in combined_df.columns if col == 'target']

        
        
        if self.convert_target :
            # First z-score normalize by date
            combined_df['target'] = combined_df.groupby(level='date')['target'].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            
            
        
        # Convert MultiIndex to single date index by dropping product level
        combined_df = combined_df.reset_index('product', drop=True)
        return combined_df 
        
        
    
    def _scale_features(self, X, fit=False):
        if self.method_scale == 'z_score':
            if fit:
                self.train_mean_ = X.mean()
                self.train_std_ = X.std()
                # Handle zero/small standard deviations
                self.train_std_ = self.train_std_.replace(0, 1)
                self.train_std_[self.train_std_ < 1e-8] = 1
            
            X_scaled = X.copy()
            for column in X.columns:
                X_scaled[column] = (X[column] - self.train_mean_[column]) / self.train_std_[column]
            
        elif self.method_scale == 'pct_change':
            X_scaled = X.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method_scale}")
            
        return X_scaled
    
    def fit(self, product_to_df_feature, product_to_target):
        """
        Fit the model using product features and targets
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
        product_to_target : dict
            Dictionary mapping products to their target DataFrames
        """
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)
        
        
        
        X = combined_df[self.feature_names_] 
        y = combined_df[self.target_names_]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        
        # Fit the model
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, product_to_df_feature):
        """
        Predict using the fitted model
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
            
        Returns:
        --------
        date_to_product_score : dict
            Dictionary with dates as keys and nested dictionaries as values,
            where nested dictionaries map products to their predictions
        """
        # Create features DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(), axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        product_to_pred = {}
        for product, df_feature in product_to_df_feature.items():
            X = df_feature.dropna().copy(deep=True)
            X_scaled = self._scale_features(X, fit=False)
            product_to_pred[product] = pd.Series(self.model.predict(X_scaled).flatten(), index=X.index)
            
        # Apply smoothing if specified
        
        
        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        
        
        # Create predictions for each date 
        for product, y_pred in product_to_pred.items():
            for date in y_pred.index:
                if date not in date_to_product_score:  # Fixed: Check if date exists
                    date_to_product_score[date] = {}   # Fixed: Initialize empty dict for new date
                date_to_product_score[date][product] = y_pred.loc[date]  # Add prediction for product

        # Apply z-score normalization if convert_target is True 
        for date, product_scores in date_to_product_score.items():
            if self.convert_target:
                scores = np.array(list(product_scores.values()))
                mean = np.mean(scores)
                std = np.std(scores)
                if std > 0:  # Avoid division by zero
                    for product in product_scores:
                        product_scores[product] = (product_scores[product] - mean) / std 
                date_to_product_score[date] = product_scores
        
        return date_to_product_score 

class Average(BaseEstimator):
    def __init__(self,smooth=0, models = []):
        self.id_to_model = {i : model for i,model in enumerate(models)} 
        
        
    
    
    
    def fit(self, product_to_df_feature, product_to_target):
        """
        Fit the model using product features and targets
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
        product_to_target : dict
            Dictionary mapping products to their target DataFrames
        """
        self.id_to_fit_model = {}
        for id , model in self.id_to_model.items():        
            self.id_to_fit_model[id] = model.fit(product_to_df_feature, product_to_target)
        return self
        
        
    
    def predict(self, product_to_df_feature):
        """
        Predict using the fitted model
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
            
        Returns:
        --------
        date_to_product_score : dict
            Dictionary with dates as keys and nested dictionaries as values,
            where nested dictionaries map products to their predictions 
        """
        id_to_date_product_score = {}
        for id, model in self.id_to_model.items():
            id_to_date_product_score[id] = model.predict(product_to_df_feature)
        
        # Collect all unique dates and products
        all_dates = set()
        all_products = set()
        for date_product_score in id_to_date_product_score.values():
            all_dates.update(date_product_score.keys())
            for product_scores in date_product_score.values():
                all_products.update(product_scores.keys())
        
        # Create averaged predictions
        date_to_product_score = {}
        for date in all_dates:
            product_scores = {}
            for product in all_products:
                # Collect all available predictions for this date and product
                scores = []
                for id in id_to_date_product_score:
                    if (date in id_to_date_product_score[id] and 
                        product in id_to_date_product_score[id][date]):
                        scores.append(id_to_date_product_score[id][date][product])
                
                # Only include product if we have at least one prediction
                if scores:
                    product_scores[product] = np.mean(scores)
            
            # Only include date if we have predictions for any products
            if product_scores:
                date_to_product_score[date] = product_scores
        
        return date_to_product_score 
    

class XGB_one_for_each(BaseEstimator):
    def __init__(self,convert_target = False,  **xgb_params):
        self.xgb_params = xgb_params
        self.convert_target = convert_target 
        self.product_to_model = {}
        self.product_to_scale = {}
    def fit(self, product_to_df_feature, product_to_target):
        """
        Fit the model using product features and targets
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
        product_to_target : dict
            Dictionary mapping products to their target DataFrames
        """
        # Create combined DataFrame
        df_target = pd.concat(product_to_target.values(), axis=1) 
        df_target.columns = list(product_to_target.keys()) 
        if self.convert_target:
            row_means = df_target.mean(axis=1)
            row_stds = df_target.std(axis=1)
            df_target = df_target.sub(row_means, axis=0).div(row_stds, axis=0)
        
        for product, df_feature in product_to_df_feature.items():
            df = pd.concat([df_feature, df_target[product]], axis=1)
            df = ut.sort_date_index(df)
            df = df.dropna()
            df.columns = list(df_feature.columns) + ['target']
            X = df.drop('target', axis=1)
            y = df['target']

            self.product_to_model[product] = xgb.XGBRegressor(**self.xgb_params)
            self.product_to_model[product].fit(X, y)  # Removed scaling
        return self 
    def predict(self, product_to_df_feature):
        """
        Predict using the fitted model
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
            
        Returns:
        --------
        date_to_product_score : dict
            Dictionary with dates as keys and nested dictionaries as values,
            where nested dictionaries map products to their predictions
        """
        product_to_pred = {}
        for product, df_feature in product_to_df_feature.items():
            X = df_feature.dropna().copy(deep=True)
            product_to_pred[product] = pd.Series(
                self.product_to_model[product].predict(X).flatten(), 
                index=X.index
            )
            
        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        for date in set().union(*[pred.index for pred in product_to_pred.values()]):
            product_scores = {}
            for product, predictions in product_to_pred.items():
                if date in predictions.index and not pd.isna(predictions[date]):
                    product_scores[product] = predictions[date]
            
            # Apply z-score normalization if convert_target is True
            if self.convert_target and product_scores:
                scores = np.array(list(product_scores.values()))
                mean = np.mean(scores)
                std = np.std(scores)
                if std > 0:  # Avoid division by zero
                    for product in product_scores:
                        product_scores[product] = (product_scores[product] - mean) / std
            
            date_to_product_score[date] = product_scores
        
        return date_to_product_score 

class XGB_one_for_all(BaseEstimator):
    def __init__(self, convert_target=False, feature_neutralization=False, **xgb_params):
        self.convert_target = convert_target
        self.feature_neutralization = feature_neutralization
        self.xgb_params = xgb_params
        self.model = xgb.XGBRegressor(**xgb_params)
        
    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")
        # Validate that all feature DataFrames have the same columns
        feature_columns = None 
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg) 
            df = pd.concat([df_feature, product_to_target[product]], axis=1)
            df.columns = list(df_feature.columns) + ['target']
            df = ut.sort_date_index(df)
            df = df.dropna()
            df['product'] = product
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=0) 
        combined_df = ut.sort_date_index(combined_df)
        # Set MultiIndex of date and product
        combined_df.set_index('product', append=True, inplace=True)
        combined_df = combined_df.reorder_levels(['date', 'product'])  # Swap to have (date, product) order
        combined_df = combined_df.sort_index(level=0)  # Sort only by date level
        print(f"Combined DataFrame index levels: {combined_df.index.names}")  # Verify index structure
        if self.convert_target : 
            # Z-score normalize only the target column by date
            row_means = combined_df.groupby(level='date')['target'].transform('mean')
            row_stds = combined_df.groupby(level='date')['target'].transform('std')
            combined_df['target'] = (combined_df['target'] - row_means) / row_stds
        # Ensure target column is named consistently
        
        
        return combined_df.dropna()
    
    def fit(self, product_to_df_feature, product_to_target, **kwargs):
        """
        Fit the model using product features and targets
        """
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)
        X = combined_df.drop('target', axis=1)
        y = combined_df['target']
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        if self.feature_neutralization > 0:
            index = y.index
            y = ut_ml.neutralize(X, y, proportion=self.feature_neutralization)
            y.index = index
            
        # Use the XGBoost model instance instead of parent class
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, product_to_df_feature, **kwargs):
        """
        Predict using the fitted model
        """
        # Create DMatrix for prediction
        feature_columns = None 
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg) 
            df = ut.sort_date_index(df_feature)
            df = df.dropna()
            df['product'] = product
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=0)
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.set_index('product', append=True)
        combined_df = combined_df.reorder_levels(['date', 'product'])
        combined_df = combined_df.sort_index(level=0)

        
        
        # Make predictions
        predictions = ut_ml.keep_multiindex_prediction(combined_df, self.model)
       
        
        if self.feature_neutralization > 0:
            index = predictions.index
            predictions = ut_ml.neutralize(
                combined_df.reset_index(drop=True),
                predictions.reset_index(drop=True),
                proportion=self.feature_neutralization
            )
            predictions.index = index

        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        for date in predictions.index.get_level_values(0).unique():
            product_scores = {
                product: predictions.loc[date, product] 
                for product in predictions.loc[date].index
            }
            date_to_product_score[date] = product_scores
        
        return dict(sorted(date_to_product_score.items(), key=lambda x: x[0]))
class XGB_one_for_all_sharpe_loss(BaseEstimator):
    def __init__(self, convert_target=False, feature_neutralization=False, **xgb_params):
        self.xgb_params = xgb_params
        self.convert_target = convert_target
        self.feature_neutralization = feature_neutralization
        self.model = None
        
    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")
        # Validate that all feature DataFrames have the same columns
        feature_columns = None 
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg) 
            df = pd.concat([df_feature, product_to_target[product]], axis=1)
            df.columns = list(df_feature.columns) + ['target']
            df = ut.sort_date_index(df)
            df = df.dropna()
            df['product'] = product
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=0) 
        combined_df = ut.sort_date_index(combined_df)
        # Set MultiIndex of date and product
        combined_df.set_index('product', append=True, inplace=True)
        combined_df = combined_df.reorder_levels(['date', 'product'])  # Swap to have (date, product) order
        combined_df = combined_df.sort_index(level=0)  # Sort only by date level
        print(f"Combined DataFrame index levels: {combined_df.index.names}")  # Verify index structure
        
        # Ensure target column is named consistently
        
        
        return combined_df.dropna()
    
    def fit(self, product_to_df_feature, product_to_target, **kwargs):
        """
        Fit the model using product features and targets
        """
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)

        X = combined_df.drop('target', axis=1)
        y = combined_df['target']
        
        if self.feature_neutralization > 0:
            index = y.index
            y = ut_ml.neutralize(X, y, proportion=self.feature_neutralization)
            y.index = index 
        group_size = combined_df.groupby(level='date').size().values 

        # Get date indices for grouping
        
        
        # Create DMatrix with date information
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group_size)
        
        # Set up parameters
        params = {
            **self.xgb_params
        }
        
        # Create a partial function for the objective with temperature
        obj_function = partial(cl.sharpe_ratio_rmse_one_for_all, temperature=0.05)
        
        # Train model with custom objective passed separately
        self.model = xgb.train(
            params,
            dtrain,
            obj=obj_function,  # Pass the objective function here
            **kwargs
        )
        
        return self

    def predict(self, product_to_df_feature, **kwargs):
        """
        Predict using the fitted model
        """
        # Create DMatrix for prediction
        feature_columns = None 
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            print(product)
            print(df_feature.head(5))
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg) 
            df = ut.sort_date_index(df_feature)
            df = df.dropna()
            df['product'] = product
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=0)
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.set_index('product', append=True)
        combined_df = combined_df.reorder_levels(['date', 'product'])
        combined_df = combined_df.sort_index(level=0)
        dtest = xgb.DMatrix(combined_df)
        
        # Make predictions
        predictions = pd.Series(
            self.model.predict(dtest, **kwargs),
            index=combined_df.index
        )
        
        if self.feature_neutralization > 0:
            index = predictions.index
            predictions = ut_ml.neutralize(
                combined_df.reset_index(drop=True),
                predictions.reset_index(drop=True),
                proportion=self.feature_neutralization
            )
            predictions.index = index

        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        for date in predictions.index.get_level_values(0).unique():
            product_scores = {
                product: predictions.loc[date, product] 
                for product in predictions.loc[date].index
            }
            date_to_product_score[date] = product_scores
        
        return dict(sorted(date_to_product_score.items(), key=lambda x: x[0]))
class XGB_multioutput(BaseEstimator):
    def __init__(self, convert_target=False,obj=None, **xgb_params):
        self.xgb_params = xgb_params
        self.convert_target = convert_target
        self.model = None 
        self.obj = obj
        
    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")
        # Validate that all feature DataFrames have the same columns
        feature_columns = None 
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg) 
            df = pd.concat([df_feature, product_to_target[product]], axis=1)
            df.columns = [f'{column}_{product}' for column in df_feature.columns] + [f'target_{product}']
            df = ut.sort_date_index(df)
            df = df.dropna()
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=1) 
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.dropna()

        target_columns = [col for col in combined_df.columns if col.startswith('target_')]
        features_columns = [col for col in combined_df.columns if col not in target_columns]
        self.target_columns = target_columns
        self.features_columns = features_columns 
        self.products = [column.replace('target_', '') for column in target_columns]
        if self.convert_target : 
            mean_target = combined_df[target_columns].mean(axis=1)
            std_target = combined_df[target_columns].std(axis=1)
            std_target = std_target.replace(0, 1)
            std_target[std_target < 1e-8] = 1
            for target_feature in target_columns : 
                combined_df[target_feature] = (combined_df[target_feature] - mean_target) / std_target
        
        
        return combined_df.dropna()
    
    def fit(self, product_to_df_feature, product_to_target, **kwargs):
        """
        Fit the model using product features and targets
        """
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)

        X = combined_df.drop(self.target_columns, axis=1)
        y = combined_df[self.target_columns]
        

        # Get date indices for grouping
        
        
        # Create DMatrix with date information
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set up parameters
        params = {
            **self.xgb_params 
        }
        params['tree_method'] = 'hist'
        params['num_target'] = len(self.target_columns)
        if self.obj is None : 
            self.model = xgb.train(
                params,
                dtrain,
                **kwargs 
            )
        else : 
            self.model = xgb.train(
                params,
                dtrain,
                obj=self.obj,
                **kwargs 
            )
        return self

    def predict(self, product_to_df_feature, **kwargs):
        """
        Predict using the fitted model
        """
        # Create DMatrix for prediction
        feature_columns = None 
        dfs = []
        for product, df_feature in product_to_df_feature.items():
            print(product)
            print(df_feature.head(5))
            if feature_columns is None:
                feature_columns = set(df_feature.columns)
            else:
                current_columns = set(df_feature.columns)
                if current_columns != feature_columns:
                    missing = feature_columns - current_columns
                    extra = current_columns - feature_columns
                    error_msg = f"Feature mismatch for product {product}.\n"
                    if missing:
                        error_msg += f"Missing columns: {missing}\n"
                    if extra:
                        error_msg += f"Extra columns: {extra}"
                    raise ValueError(error_msg) 
            df = ut.sort_date_index(df_feature)
            df.columns = [f'{column}_{product}' for column in df_feature.columns]
            df = df.dropna()
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=1)
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.dropna()
        print(combined_df.head(5))
        dtest = xgb.DMatrix(combined_df)
        
        # Make predictions
        predictions = pd.DataFrame(
            self.model.predict(dtest, **kwargs),
            index=combined_df.index,  # This already contains the dates
            columns=self.products  # Use target_columns instead of products
        )
        

        # Convert predictions to date_to_product_score format
        date_to_product_score = predictions.to_dict(orient='index')
        print(date_to_product_score)

        return dict(sorted(date_to_product_score.items(), key=lambda x: x[0]))
 
    
class LinearRegression_one_for_each(BaseEstimator):
    def __init__(self, method_scale='z_score',convert_target = None,  **regression_params):
        self.method_scale = method_scale
        self.regression_params = regression_params
        self.train_mean_ = None
        self.train_std_ = None
        self.feature_names_ = None
        self.target_names_ = None
        self.convert_target = convert_target 
        self.product_to_model = {}
        self.product_to_scale = {}  
        
        
    
    def _scale_features(self, X,product, fit=False):
        if self.method_scale == 'z_score':
            if fit: 
                if product not in self.product_to_scale : 
                    self.product_to_scale[product] = {}
                self.product_to_scale[product]['train_mean'] = X.mean()
                self.product_to_scale[product]['train_std'] = X.std()
                # Handle zero/small standard deviations
                self.product_to_scale[product]['train_std'] = self.product_to_scale[product]['train_std'].replace(0, 1)
                self.product_to_scale[product]['train_std'][self.product_to_scale[product]['train_std'] < 1e-8] = 1
            
            X_scaled = X.copy()
            for column in X.columns:
                X_scaled[column] = (X[column] - self.product_to_scale[product]['train_mean'][column]) / self.product_to_scale[product]['train_std'][column]
            
        elif self.method_scale == 'pct_change':
            X_scaled = X.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method_scale}")
            
        return X_scaled
    
    def fit(self, product_to_df_feature, product_to_target):
        """
        Fit the model using product features and targets
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
        product_to_target : dict
            Dictionary mapping products to their target DataFrames
        """
        # Create combined DataFrame
        df_target = pd.concat(product_to_target.values(), axis=1) 
        df_target.columns = list(product_to_target.keys()) 
        if self.convert_target:
            row_means = df_target.mean(axis=1)
            row_stds = df_target.std(axis=1)
            df_target = df_target.sub(row_means, axis=0).div(row_stds, axis=0)
        
        for product, df_feature in product_to_df_feature.items():
            df = pd.concat([df_feature,df_target[product]],axis=1)
            df = ut.sort_date_index(df)
            df = df.dropna()
            df.columns = list(df_feature.columns) + ['target']
            X = df.drop('target',axis=1)
            y = df['target']

            X_scaled = self._scale_features(X,product,fit=True) 
            self.product_to_model[product] = LinearRegression(**self.regression_params)
            self.product_to_model[product].fit(X_scaled,y)
        return self
        
        
    
    def predict(self, product_to_df_feature):
        """
        Predict using the fitted model
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
            
        Returns:
        --------
        date_to_product_score : dict
            Dictionary with dates as keys and nested dictionaries as values,
            where nested dictionaries map products to their predictions
        """
        # Create features DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(), axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        product_to_pred = {}
        for product, df_feature in product_to_df_feature.items():
            X = df_feature.dropna().copy(deep=True)
            X_scaled = self._scale_features(X, product, fit=False)
            product_to_pred[product] = pd.Series(self.product_to_model[product].predict(X_scaled).flatten(), index=X.index)

        
            
        df_pred = pd.concat(product_to_pred.values(), axis=1)
        df_pred.columns = list(product_to_pred.keys())
        
        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        all_dates = set()
        # First collect all dates from all products
        for product in product_to_pred:
            all_dates.update(df_pred[product].index)
        
        # Create predictions for each date
        for date in all_dates:
            product_scores = {}
            for product in df_pred.columns:
                # Only include non-NaN values
                if date in df_pred[product].index and not pd.isna(df_pred.loc[date, product]):
                    score = df_pred.loc[date, product]
                    
                    product_scores[product] = score
            # Apply z-score normalization if convert_target is True
            if self.convert_target and product_scores:
                scores = np.array(list(product_scores.values()))
                mean = np.mean(scores)
                std = np.std(scores)
                if std > 0:  # Avoid division by zero
                    for product in product_scores:
                        product_scores[product] = (product_scores[product] - mean) / std
            
            date_to_product_score[date] = product_scores
            
        return date_to_product_score 
    
class PLS_Regressor_each_feature(BaseEstimator):
    def __init__(self, n_components=2, method_scale='z_score', convert_target=None, **pls_params):
        self.n_components = n_components
        self.method_scale = method_scale
        self.pls_params = pls_params
        self.model = PLSRegression(n_components=n_components)
        self.train_mean_ = None
        self.train_std_ = None
        self.feature_names_ = None
        self.target_names_ = None
        self.convert_target = convert_target
        
    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")

        # Create combined DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(),axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        
        targets_df = pd.concat(product_to_target.values(), keys=product_to_target.keys(),axis=1)
        targets_df.columns = [f"{product}_target" for product in product_to_df_feature.keys()]
        
        # Apply z-score normalization to targets if specified
        if self.convert_target :
            # Z-score normalize each row
            row_means = targets_df.mean(axis=1)
            row_stds = targets_df.std(axis=1)
            targets_df = targets_df.sub(row_means, axis=0).div(row_stds, axis=0)
            
            
        
        combined_df = pd.concat([features_df, targets_df], axis=1)
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.dropna()
        return combined_df.dropna()
    
    def _scale_features(self, X, fit=False):
        if self.method_scale == 'z_score':
            if fit:
                self.train_mean_ = X.mean()
                self.train_std_ = X.std()
                # Handle zero/small standard deviations
                self.train_std_ = self.train_std_.replace(0, 1)
                self.train_std_[self.train_std_ < 1e-8] = 1
            
            X_scaled = X.copy()
            for column in X.columns:
                X_scaled[column] = (X[column] - self.train_mean_[column]) / self.train_std_[column]
            
        elif self.method_scale == 'pct_change':
            X_scaled = X.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method_scale}")
            
        return X_scaled
    
    def fit(self, product_to_df_feature, product_to_target):
        """
        Fit the model using product features and targets
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
        product_to_target : dict
            Dictionary mapping products to their target DataFrames
        """
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)
        
        # Split features and targets
        self.feature_names_ = [col for col in combined_df.columns if not col.endswith('_target')]
        self.target_names_ = [col for col in combined_df.columns if col.endswith('_target')]
        
        X = combined_df[self.feature_names_]
        y = combined_df[self.target_names_]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, product_to_df_feature):
        """
        Predict using the fitted model
        
        Parameters:
        -----------
        product_to_df_feature : dict
            Dictionary mapping products to their feature DataFrames
            
        Returns:
        --------
        date_to_product_score : dict
            Dictionary with dates as keys and nested dictionaries as values,
            where nested dictionaries map products to their predictions
        """
        # Create features DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(),axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names_) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = features_df[self.feature_names_]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=False)
        
        # Make predictions
        y_pred = pd.DataFrame(
            self.model.predict(X_scaled),
            index=X.index,
            columns=self.target_names_
        )

        # Z-score normalize each row 
        if self.convert_target :
            row_means = y_pred.mean(axis=1)
            row_stds = y_pred.std(axis=1)
            y_pred = y_pred.sub(row_means, axis=0).div(row_stds, axis=0)
        
         
        # Apply smoothing if specified
        
        
        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        for date in y_pred.index:
            product_scores = {}
            # First pass: apply tanh if needed
            for col in y_pred.columns:
                product = col.replace('_target', '')
                score = y_pred.loc[date, col]
                product_scores[product] = score 
            date_to_product_score[date] = product_scores 
            
            
            
        return date_to_product_score
        
        

class PLS_Regressor_each_feature_with_xgb(BaseEstimator):
    def __init__(self, n_components=2, method_scale='z_score', convert_target=None, **xgb_params):
        self.n_components = n_components
        self.method_scale = method_scale
        self.xgb_params = xgb_params
        self.model = PLSRegression(n_components=n_components)
        self.train_mean_ = None
        self.train_std_ = None
        self.feature_names_ = None
        self.target_names_ = None
        self.convert_target = convert_target
        self.target_to_xgboost = {}

    def _create_combined_df(self, product_to_df_feature, product_to_target):
        # Validate that both dictionaries have the same keys
        if set(product_to_df_feature.keys()) != set(product_to_target.keys()):
            raise ValueError("Product mismatch between features and targets dictionaries")

        # Create combined DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(), axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        
        targets_df = pd.concat(product_to_target.values(), keys=product_to_target.keys(), axis=1)
        targets_df.columns = [f"{product}_target" for product in product_to_df_feature.keys()]
        
        # Apply z-score normalization to targets if specified
        if self.convert_target:
            # Z-score normalize each row
            row_means = targets_df.mean(axis=1)
            row_stds = targets_df.std(axis=1)
            targets_df = targets_df.sub(row_means, axis=0).div(row_stds, axis=0)
        
        combined_df = pd.concat([features_df, targets_df], axis=1)
        combined_df = ut.sort_date_index(combined_df)
        combined_df = combined_df.dropna()
        return combined_df

    def _scale_features(self, X, fit=False):
        if self.method_scale == 'z_score':
            if fit:
                self.train_mean_ = X.mean()
                self.train_std_ = X.std()
                # Handle zero/small standard deviations
                self.train_std_ = self.train_std_.replace(0, 1)
                self.train_std_[self.train_std_ < 1e-8] = 1
            
            X_scaled = X.copy()
            for column in X.columns:
                X_scaled[column] = (X[column] - self.train_mean_[column]) / self.train_std_[column]
            
        elif self.method_scale == 'pct_change':
            X_scaled = X.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method_scale}")
            
        return X_scaled

    def fit(self, product_to_df_feature, product_to_target):
        # Create combined DataFrame
        combined_df = self._create_combined_df(product_to_df_feature, product_to_target)
        
        # Split features and targets
        self.feature_names_ = [col for col in combined_df.columns if not col.endswith('_target')]
        self.target_names_ = [col for col in combined_df.columns if col.endswith('_target')]
        
        X = combined_df[self.feature_names_]
        y = combined_df[self.target_names_]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        available_dates = X_scaled.index
        
        # Fit PLS model
        self.model.fit(X_scaled, y)
        
        # Fit XGBoost models for each target
        for target in self.target_names_:
            product = target.replace('_target', '')
            X_train_xgboost = product_to_df_feature[product].loc[available_dates]
            # Get the column index for the current target
            target_idx = self.target_names_.index(target)
            X_train_xgboost['PLS_target'] = self.model.predict(X_scaled)[:, target_idx]
            X_train_xgboost = ut.sort_date_index(X_train_xgboost)

            model_xgboost = xgb.XGBRegressor(**self.xgb_params)
            model_xgboost.fit(X_train_xgboost, y[target])
            self.target_to_xgboost[target] = model_xgboost
        
        return self

    def predict(self, product_to_df_feature):
        # Create features DataFrame
        features_df = pd.concat(product_to_df_feature.values(), keys=product_to_df_feature.keys(), axis=1)
        features_df.columns = [f"{product}_{col}" for product, col in features_df.columns]
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names_) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = features_df[self.feature_names_]
        X_scaled = self._scale_features(X, fit=False)
        
        # Get PLS predictions
        pls_predictions = pd.DataFrame(
            self.model.predict(X_scaled),
            index=X.index,
            columns=self.target_names_
        )
        
        # Get XGBoost predictions for each target
        xgb_predictions = {}
        for target, xgb_model in self.target_to_xgboost.items():
            product = target.replace('_target', '')
            if product not in product_to_df_feature:
                continue
                
            X_xgb = product_to_df_feature[product].copy()
            X_xgb['PLS_target'] = pls_predictions[target]
            xgb_predictions[target] = pd.Series(
                xgb_model.predict(X_xgb),
                index=X_xgb.index
            )

        # Apply smoothing if specified
        

        # Convert predictions to date_to_product_score format
        date_to_product_score = {}
        for date in pls_predictions.index:
            product_scores = {}
            for target in self.target_names_:
                product = target.replace('_target', '')
                if target in xgb_predictions and date in xgb_predictions[target].index:
                    score = xgb_predictions[target][date]
                    product_scores[product] = score

            # Z-score normalize the scores for this date if specified
            if self.convert_target and product_scores:
                scores = np.array(list(product_scores.values()))
                mean = np.mean(scores)
                std = np.std(scores)
                if std > 0:  # Avoid division by zero
                    for product in product_scores:
                        product_scores[product] = (product_scores[product] - mean) / std
            
            date_to_product_score[date] = product_scores
        
        return date_to_product_score