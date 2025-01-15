import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import os  
import concurrent 
import copy 
import json 
import math

import adv_machine.get_prices as gp 
import adv_machine.feature_computer as ft 
  
import adv_machine.config as cf 
from adv_machine.log import _print   
import adv_machine.utils as ut  
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Move the compute_features function outside the class
def compute_features(products_chunk, config_features):
    results = []
    for product in products_chunk:
        try:
            # Create configs for this product
            configs = []
            for config in config_features:
                config_copy = copy.deepcopy(config)
                key = list(config_copy.keys())[0]
                config_copy[key]['collect']['product'] = product
                configs.append(config_copy)
                
            df_feature = ft.feature_computer(configs, verbose=3)
            if df_feature.empty : 
                continue 
            else : 
                series_feature = df_feature.mean(axis=1)
                series_feature.name = product
                results.append(series_feature)
        except Exception as e:
            print(f"Error processing {product}: {str(e)}")
            continue
    return results

class Universe():
    def __init__(self,baseline_universe, config_universe,n_jobs=-1,verbose=0):
        self.products = get_baseline_universe(baseline_universe)
        self.baseline_universe = baseline_universe  
        self.config_universe = config_universe 
        self.n_jobs = n_jobs
        self.verbose = verbose 
       
    def compute_universe(self,end_date):
        # Try to load existing universe from JSON first
        top = self.config_universe['top']

        json_path = f'universe/rolling_universe_{self.baseline_universe}.json'
        end_date_datetime = ut.format_datetime(pd.to_datetime(end_date))

        config_features = self.config_universe.get('config_features',[])
        
        if os.path.exists(json_path):
            _print(f'Loading universe from {json_path}', 1, self.verbose)
            # Load existing universe from JSON
            with open(json_path, 'r') as f:
                date_to_product_values = json.load(f)
            # Convert string 'NaN' back to actual NaN values
            date_to_product_values = {
                ut.format_datetime(pd.to_datetime(date)): {
                    product: np.nan if value == 'NaN' else value 
                    for product, value in product_to_value.items()
                }
                for date, product_to_value in date_to_product_values.items()
            }
            last_date = list(date_to_product_values.keys())[-1]
            if last_date >= end_date_datetime :
                _print(f'Universe already computed for {end_date}', 1, self.verbose)
                date_to_product_values = {date : product_to_value for date, product_to_value in date_to_product_values.items() if date <= end_date_datetime}
                date_to_product_universe = mark_top_rank(date_to_product_values,top,verbose = self.verbose) 

                all_products = set()
                for product_list in date_to_product_universe.values():
                    all_products.update(product_list)
                
                self.active_universe = list(all_products)
                self.date_to_product_universe = date_to_product_universe
                self.date_to_product_values = date_to_product_values 
                return 
                
            else : 
                _print(f'Universe is available but not up to date. Last date is {last_date}', 1, self.verbose)
        
        
        
        if not config_features :
            _print('Computing universe based on prices', 1, self.verbose)
            concat = []
            for product in tqdm(self.products,total = len(self.products),desc = 'Computing prices'):
                series_prices = gp.get_stock_ohlc(product)['close']
                series_prices.name = product
                concat.append(series_prices)
            df = pd.concat(concat,axis=1)
            df = ut.sort_date_index(df)
            date_to_product_values = df.to_dict(orient = 'index')
            
            
            
            
        else :
            # Simplified parallel processing logic
            if self.n_jobs == -1:
                max_workers = min(os.cpu_count() - 1, 8)  # Leave one CPU free
            else:
                max_workers = self.n_jobs

            # Calculate optimal chunk size
            chunk_size = max(1, len(self.products) // (max_workers * 4))
            product_chunks = [self.products[i:i + chunk_size] 
                            for i in range(0, len(self.products), chunk_size)]

            _print(f'Processing {len(product_chunks)} chunks with {max_workers} workers', 
                  1, self.verbose)

            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks at once
                future_to_chunk = {
                    executor.submit(compute_features, chunk, config_features): i 
                    for i, chunk in enumerate(product_chunks)
                }

                # Process results as they complete
                for future in tqdm(as_completed(future_to_chunk), 
                                 total=len(future_to_chunk),
                                 desc='Processing chunks'):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_result = future.result(timeout=300)  # 5 minute timeout
                        results.extend(chunk_result)
                        _print(f'Completed chunk {chunk_idx + 1}/{len(product_chunks)}', 
                              2, self.verbose)
                    except Exception as e:
                        _print(f"Error in chunk {chunk_idx}: {str(e)}", 1, self.verbose)
                        continue

            # Process results
            if not results:
                raise RuntimeError("No results were successfully processed")

            df = pd.concat(results, axis=1)
            df = ut.sort_date_index(df)
            date_to_product_values = df.to_dict(orient='index')
            
        date_to_product_values_store = {
            date.strftime('%Y-%m-%d'): {
                product: 'NaN' if pd.isna(value) else value 
                for product, value in product_to_value.items()
            }
            for date, product_to_value in date_to_product_values.items()
        }

        # Save the updated universe to JSON
        _print(f'Saving universe to {json_path}', 1, self.verbose)
        with open(json_path, 'w') as f:
            json.dump(date_to_product_values_store, f)
        
        
        

        _print(f'Computing top {top} products for each date', 1, self.verbose)
        date_to_product_values = {date : product_to_value for date, product_to_value in date_to_product_values.items() if date <= end_date_datetime}
        date_to_product_universe = mark_top_rank(date_to_product_values,top,verbose = self.verbose)
        
        all_products = set()
        for product_list in date_to_product_universe.values():
            all_products.update(product_list)
        
        _print(f'Active universe contains {len(all_products)} products', 1, self.verbose)
        self.active_universe = list(all_products)
        self.date_to_product_universe = date_to_product_universe
        self.date_to_product_values = date_to_product_values 


 
            
            

def mark_top_rank(date_to_product_values, top, verbose=0):
    """
    Mark the top X values for each date using pandas rank functionality.
    
    Args:
        date_to_product_values (dict): Dictionary where:
            - keys are dates
            - values are dictionaries (product -> ranking value)
        top (int): Number of top values to mark in each date
        verbose (int): Verbosity level for printing
    """
    
    products_size = len(list(date_to_product_values.values())[0]) if top is None else top 
    _print(f"Processing {len(date_to_product_values)} dates with {products_size} products each", 1, verbose)
    
    if top is None:
        _print("No top limit specified, including all non-NaN values", 1, verbose)
        result = {date : {product : 1-int(pd.isna(value)) for product, value in product_values.items()} 
                 for date, product_values in date_to_product_values.items()}
        result = {date : [product for product, value in product_values.items() if value == 1] 
                 for date, product_values in result.items()}
    else:
        _print(f"Selecting top {top} products for each date", 1, verbose)
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame.from_dict(date_to_product_values, orient='index')
        
        # Calculate ranks (ascending=False means highest values get rank 1)
        ranks = df.rank(axis=1, method='first', ascending=False)
        
        # Create mask for top products (rank <= top)
        top_mask = ranks <= top
        
        # Convert back to dictionary format
        result = {
            date: list(products[mask].index)
            for (date, products), (_, mask) in zip(df.iterrows(), top_mask.iterrows())
        }

    # Find first date with complete data
    first_date = None
    for date, product_list in result.items():
        if len(product_list) == products_size:
            first_date = date
            break
    
    if first_date is not None:
        _print(f"Found first complete date: {first_date}", 1, verbose)
        original_len = len(result)
        result = {date : product_list for date, product_list in result.items() if date >= first_date}
        _print(f"Filtered from {original_len} to {len(result)} dates", 1, verbose)
            
    return result


def get_baseline_universe(baseline_universe):
    if baseline_universe == 'available_stock':
        baseline_universe = cf.available_stocks 
    elif baseline_universe == 'us_stock':
        baseline_universe = cf.us_stock 
    else : 
        raise ValueError(f"Invalid baseline universe type: {type(baseline_universe)}")
    return baseline_universe
    

    


    

















 



