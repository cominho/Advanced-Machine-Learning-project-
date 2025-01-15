from adv_machine.log import _print, _print_error  


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
                pnl_date_product = quantity * (close_price - entry_price)
            else:
                entry_price = date_to_product_prices[date_open][product_name]["open"] * (1 - fees_bps * 1e-4)
                close_price = date_to_product_prices[date_close][product_name]["close"] * (1 + fees_bps * 1e-4)
                quantity = abs(signal) * short_allocation_fraction * (notional / nb_short) / entry_price
                pnl_date_product = -quantity * (close_price - entry_price)

            _print(f"Product: {product_name}, Signal: {signal}, Entry Price: {entry_price}, Close Price: {close_price}, PnL: {pnl_date_product}", 3, verbose)

            individuals_pnls[date_open][product_name] = {"signal": signal, "pnl": pnl_date_product}
            pnl_date += pnl_date_product

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