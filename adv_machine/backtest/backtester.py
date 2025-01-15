from adv_machine.log import _print, _print_error  


import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta 
import uuid 
from collections import defaultdict 
import plotly.graph_objects as go 
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor 
import copy          

import adv_machine.get_prices as gp 

import adv_machine.backtest.metrics as metrics 
 
import adv_machine.utils as ut


def calculate_cumulative_pnl_v1(date_to_signal, date_to_product_prices, fees_bps, notional=10000, verbose=0):
    dates = list(date_to_signal.keys())
    first_date = min(dates)
    pnls = {first_date: 0}
    pnls_long = []
    pnls_short = []
    next_date_to_execute = None 
    long_allocation_fraction = 1/2
    short_allocation_fraction = 1/2
    
    individuals_pnls = {}
    

    _print("Starting PnL calculation", 1, verbose)

    for i, date_open in enumerate(dates):
        if next_date_to_execute is not None and (dates[-1] == date_open) :
            _print(f"Do not trade on {date_open}", 2, verbose)
            continue

        date_close = dates[i+1]
        next_date_to_execute = dates[i+1]
        product_name_to_signal = date_to_signal[date_open]
        pnl_date = 0
        nb_product_names = len(product_name_to_signal)
        nb_long = 0
        nb_short = 0

        _print(f"Processing date {date_open} with signals: {product_name_to_signal}", 2, verbose)

        for product_name, signal in product_name_to_signal.items():
            if signal >= 0:
                nb_long += 1
            else:
                nb_short += 1

        _print(f"Number of long positions: {nb_long}, short positions: {nb_short}", 2, verbose)

        individuals_pnls[date_open] = {}
        for product_name, signal in product_name_to_signal.items():
            
            if signal == 0 :
                continue 
            elif signal > 0:
                entry_price = date_to_product_prices[date_open][product_name]["open"] * (1 + fees_bps * 1e-4)
                close_price = date_to_product_prices[date_close][product_name]["close"] * (1 - fees_bps * 1e-4)
                quantity = abs(signal) * long_allocation_fraction * (notional / nb_long) / entry_price
                pnl_date_crypto = quantity * (close_price - entry_price)
            else:
                entry_price = date_to_product_prices[date_open][product_name]["open"] * (1 - fees_bps * 1e-4)
                close_price = date_to_product_prices[date_close][product_name]["close"] * (1 + fees_bps * 1e-4)
                quantity = abs(signal) * short_allocation_fraction * (notional / nb_short) / entry_price
                pnl_date_crypto = -quantity * (close_price - entry_price)

            _print(f"Product: {product_name}, Signal: {signal}, Entry Price: {entry_price}, Close Price: {close_price}, PnL: {pnl_date_crypto}", 3, verbose)

            individuals_pnls[date_open][product_name] = {"signal": signal, "pnl": pnl_date_crypto}
            pnl_date += pnl_date_crypto

        pnls[date_close] = pnl_date
        _print(f"Calculated PnL for {date_open} to {date_close}: {pnl_date}", 2, verbose)

    current_pnl = 0
    cumulative_pnl = {}
    for date, pnl in pnls.items():
        current_pnl += pnl
        cumulative_pnl[date] = current_pnl
        _print(f"Cumulative PnL for {date}: {current_pnl}", 2, verbose)

    series_cumulative_pnl = pd.Series(cumulative_pnl)
    _print("Completed PnL calculation", 1, verbose)
    return series_cumulative_pnl
  
 

### V2 BACKTEST ###  


def calculate_cumulative_pnl_v2(active_universe, signal_type,date_to_products_signal, date_to_product_prices, fees_bps, notional=10000,verbose = 0):
    current_position = defaultdict(list)
    last_realized    = 0
    date_to_pnl      = {}
    for date, products_to_signal in sorted(date_to_products_signal.items()):
        target_position = get_target_position(date = date,
                                              active_universe = active_universe, 
                                              products_to_signal = products_to_signal,
                                              signal_type = signal_type,
                                              date_to_product_prices = date_to_product_prices, 
                                              fees_bps = fees_bps, 
                                              notional = notional,
                                              verbose = verbose)
        realized_pnl, unrealized_pnl, new_current = get_pnl(date =date,
                                                            products = active_universe, 
                                                            current_position = current_position, 
                                                            target_position = target_position, 
                                                            date_to_product_prices = date_to_product_prices, 
                                                            fees_bps = fees_bps,
                                                            verbose = verbose)
        date_to_pnl[date] = round(last_realized + realized_pnl + unrealized_pnl, 8)
        last_realized += realized_pnl
        current_position = new_current
    series_pnl = pd.Series(list(date_to_pnl.values()),index = list(date_to_pnl.keys()))
    return series_pnl




def get_target_position(date, active_universe, products_to_signal, signal_type, date_to_product_prices, fees_bps, notional, verbose=0):
    _print(f"Starting get_target_position for date {date}", 1, verbose)
    _print(f"Signal type: {signal_type}, Notional: {notional}, Fees bps: {fees_bps}", 2, verbose)
    
    # quantity is homogeneous to a quantity in dollars
    target_position = {p: {"quantity": 0} for p in active_universe}
    nb_long = 0
    nb_short = 0
    long_allocation_fraction = 1/2
    short_allocation_fraction = 1/2

    _print(f"Initial target position: {target_position}", 3, verbose)
    _print(f"Products to signal: {products_to_signal}", 2, verbose)

    for signal_value in products_to_signal.values():
        if signal_value > 0:
            nb_long += 1
        elif signal_value <0:
            nb_short += 1

    _print(f"Number of long signals: {nb_long}, short signals: {nb_short}", 1, verbose)
    _print(f"Long allocation fraction: {long_allocation_fraction}, Short allocation fraction: {short_allocation_fraction}", 2, verbose)

    for product_name, signal_value in products_to_signal.items():
        _print(f"Processing {product_name} with signal value: {signal_value}", 2, verbose)
        
        if signal_value != 0 : 
            if signal_type == "quantity":
                target_position[product_name] = {"quantity": signal_value}
                _print(f"Set target position for {product_name} with quantity signal: {signal_value}", 2, verbose)
                
            elif signal_type == "allocation_fraction":
                _print(f"Processing allocation_fraction for {product_name}", 2, verbose)
                try:
                    entry_price_wo_fees = date_to_product_prices[date][product_name]["open"]
                    _print(f"Entry price without fees for {product_name}: {entry_price_wo_fees}", 3, verbose)
                except KeyError:
                    _print_error(f'There is an issue with {product_name} for date {date}')
                    raise ValueError(f'There is an issue with {product_name} for date {date}')

                if signal_value > 0:
                    entry_price_with_fees = entry_price_wo_fees * (1 + fees_bps * 1e-4)
                    allocation_fraction = long_allocation_fraction
                    n = nb_long
                else:
                    entry_price_with_fees = entry_price_wo_fees * (1 - fees_bps * 1e-4)
                    allocation_fraction = short_allocation_fraction
                    n = nb_short

                _print(f"Entry price with fees: {entry_price_with_fees}, Allocation fraction: {allocation_fraction}, n: {n}", 3, verbose)
                
                quantity = signal_value * allocation_fraction * (notional / n) / entry_price_with_fees
                target_position[product_name] = {"quantity": quantity}
                _print(f"Calculated quantity for {product_name}: {quantity}", 2, verbose)
                
            elif signal_type == "signal_size":
                _print(f"Processing signal_size for {product_name}", 2, verbose)
                try:
                    entry_price_wo_fees = date_to_product_prices[date][product_name]["open"]
                    _print(f"Entry price without fees for {product_name}: {entry_price_wo_fees}", 3, verbose)
                except KeyError:
                    _print_error(f'There is an issue with {product_name} for date {date}')
                    raise ValueError(f'There is an issue with {product_name} for date {date}')
                
                if signal_value > 0 : 
                    entry_price_with_fees = entry_price_wo_fees*(1 + fees_bps*1e-4)
                else : 
                    entry_price_with_fees = entry_price_wo_fees*(1-fees_bps*1e-4)

                _print(f"Entry price with fees for {product_name}: {entry_price_with_fees}", 3, verbose)

                quantity = signal_value * notional/entry_price_with_fees 
                target_position[product_name] = {"quantity": quantity}
                _print(f"Set target position for {product_name} with signal_size signal: {signal_value}, quantity: {quantity}", 2, verbose)

    _print(f"Final target position: {target_position}", 1, verbose)
    return target_position


def get_pnl(date, products, current_position, target_position, date_to_product_prices, fees_bps, verbose=0):
    _print(f"Calculating PnL for date: {date}", 1, verbose)

    #current_position = {product : [{'quantity1': q1},{'quantity2':q2}]}
    product_name_to_opened_quantity = {p: 0 for p in products}
    for product_name, position_elements in current_position.items():
        opened_quantity = sum(position_element["quantity"] for position_element in position_elements)
        product_name_to_opened_quantity[product_name] = opened_quantity

    _print(f"Opened quantities: {product_name_to_opened_quantity}", 1, verbose)

    new_elements     = []
    new_realized_pnl = 0
    for product_name in products:

        current_position_for_product_name = current_position[product_name]
        target_position_for_product_name  = target_position.get(product_name)

        opened_quantity = product_name_to_opened_quantity[product_name]
        target_quantity = target_position[product_name]["quantity"]
        to_execute      = round(target_quantity - opened_quantity, 10)

        _print(f"Processing {product_name}: opened_quantity={opened_quantity}, target_quantity={target_quantity}, to_execute={to_execute}", 2, verbose)

        if to_execute == 0:
            continue
        # Increase
        if to_execute * opened_quantity >= 0:
            # Long to a long position or Short to a short postion 
            entry_price_wo_fees   = date_to_product_prices[date][product_name]["open"]
            fees_multiplier       = (1 + fees_bps * 1e-4) if to_execute > 0 else (1 - fees_bps * 1e-4)
            entry_price_with_fees = entry_price_wo_fees * fees_multiplier
            new_element = {
                "quantity"             : to_execute,
                "entry_price_with_fees": entry_price_with_fees,
                "trade_id"             : str(uuid.uuid4())
            }
            current_position[product_name].append(new_element)

            _print(f"Added new element for {product_name}: {new_element}", 3, verbose)

        # Decrease
        if to_execute * opened_quantity < 0:
            # Long to a short position or Short to a long position
            position_elements = current_position[product_name]
            ids_to_remove     = []
            for position_element in position_elements:
                if ut.is_null(to_execute):
                    break
                position_element_quantity = position_element["quantity"]
                assert position_element_quantity * to_execute < 0
                if abs(to_execute) >= abs(position_element_quantity):
                    id_to_remove        = position_element["trade_id"]
                    ids_to_remove.append(id_to_remove)
                    exit_price_wo_fees   = date_to_product_prices[date][product_name]["open"]
                    fees_multiplier      = (1 + fees_bps * 1e-4) if to_execute > 0 else (1 - fees_bps * 1e-4)
                    exit_price_with_fees = exit_price_wo_fees * fees_multiplier
                    new_realized_pnl    += position_element_quantity * (exit_price_with_fees - position_element["entry_price_with_fees"])
                    to_execute          += position_element_quantity

                    _print(f"Closed position for {product_name}: {position_element}, new_realized_pnl={new_realized_pnl}", 3, verbose)

                elif abs(to_execute) < abs(position_element_quantity):
                    trade_id = position_element["trade_id"]
                    exit_price_wo_fees   = date_to_product_prices[date][product_name]["open"]
                    fees_multiplier      = (1 + fees_bps * 1e-4) if to_execute > 0 else (1 - fees_bps * 1e-4)
                    exit_price_with_fees = exit_price_wo_fees * fees_multiplier
                    new_realized_pnl    += -to_execute * (exit_price_with_fees - position_element["entry_price_with_fees"])
                    position_element["quantity"] += to_execute
                    to_execute           = 0

                    _print(f"Partially closed position for {product_name}: {position_element}, new_realized_pnl={new_realized_pnl}", 3, verbose)

            position_elements = [e for e in position_elements if e["trade_id"] not in ids_to_remove]
            if not ut.is_null(to_execute):
                trade_id              = str(uuid.uuid4())
                entry_price_wo_fees   = date_to_product_prices[date][product_name]["open"]
                fees_multiplier       = (1 + fees_bps * 1e-4) if to_execute > 0 else (1 - fees_bps * 1e-4)
                entry_price_with_fees = entry_price_wo_fees * fees_multiplier
                new_element = {
                    "quantity"             : to_execute,
                    "entry_price_with_fees": entry_price_with_fees,
                    "trade_id"             : trade_id
                }
                position_elements.append(new_element)

                _print(f"Adjusted position for {product_name}: {new_element}", 3, verbose)

            current_position[product_name] = position_elements

    unrealized_pnl = 0
    for product_name, position_elements in current_position.items():
        if not position_elements:
            continue 
        # If the quantity is null we dont calculate the unrealized pnl  
        first_position_element          = position_elements[0]
        first_position_element_quantity = first_position_element["quantity"]
        if ut.is_null(first_position_element_quantity):
            continue
        mark_price_wo_fees              = date_to_product_prices[date][product_name]["open"]
        fees_multiplier                 = (1 - fees_bps * 1e-4) if first_position_element_quantity > 0 else (1 + fees_bps * 1e-4)
        mark_price_with_fees            = mark_price_wo_fees * fees_multiplier
        for position_element in position_elements:
            unrealized_pnl += position_element["quantity"] * (mark_price_with_fees - position_element["entry_price_with_fees"])

        _print(f"Unrealized PnL for {product_name}: {unrealized_pnl}", 2, verbose)

    return new_realized_pnl, unrealized_pnl, current_position 



def calculate_cumulative_returns_from_cumulative_pnl(series_cumulative_pnl, initial_capital) :
    series_cumulative_returns = series_cumulative_pnl.apply(lambda x : 100*((initial_capital+x)/initial_capital -1))
    return series_cumulative_returns 

def get_cumulative_pnl_benchmark(benchmark,date_to_product_signal,fees_bps,notional=10000,version = 'v1'):
    date_to_ohlc_benchmark = gp.get_prices(benchmark).to_dict(orient='index')
    date_to_product_prices_benchmark = {date : {} for date in date_to_ohlc_benchmark.keys()}
    for date in date_to_ohlc_benchmark.keys():
        date_to_product_prices_benchmark[date][benchmark] = date_to_ohlc_benchmark[date]
    date_to_product_signal_benchmark = {date : {benchmark : 1} for date in date_to_product_prices_benchmark.keys() if date in date_to_product_signal.keys() }
    if version == 'v1':
        series_cumulative_pnl_benchmark = calculate_cumulative_pnl_v1(date_to_signal =date_to_product_signal_benchmark,
                                                                         date_to_product_prices =date_to_product_prices_benchmark,
                                                                         fees_bps=fees_bps,
                                                                         notional=notional)
    elif version == 'v2' : 
        series_cumulative_pnl_benchmark = calculate_cumulative_pnl_v2(active_universe=[benchmark],
                                                                      signal_type='allocation_fraction',
                                                                      date_to_products_signal=date_to_product_signal_benchmark,
                                                                      date_to_product_prices=date_to_product_prices_benchmark,
                                                                      fees_bps=fees_bps,
                                                                      notional=notional)
    return series_cumulative_pnl_benchmark