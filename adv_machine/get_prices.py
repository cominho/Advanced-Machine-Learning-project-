import yfinance as yf
import pandas as pd

def get_stock_ohlc(stock_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Retrieve OHLC and volume data for a stock using Yahoo Finance.

    Parameters:
        stock_name (str): The ticker symbol of the stock.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing OHLC and volume data with a datetime index.
    """
    try:
        # Add progress=False to avoid multi-level columns
        stock_data = yf.download(stock_name, start=start_date, end=end_date, progress=False)
        
        # If we still get multi-level columns, flatten them
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
            
        ohlc_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        ohlc_data.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlc_data.index = pd.to_datetime(ohlc_data.index)
        
        # Find the first non-zero volume index
        first_valid_volume = ohlc_data[ohlc_data['volume'] > 0].index[0]
        first_valid_price = ohlc_data[ohlc_data['close'] > 0].index[0]
        first_valid_idx = max(first_valid_volume,first_valid_price)
        
        # Trim the DataFrame to start from first non-zero volume
        ohlc_data = ohlc_data[first_valid_idx:]
        
        return ohlc_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    
def get_bulk_ohlc(stock_names: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """
    Retrieve OHLC and volume data for multiple stocks using Yahoo Finance.

    Parameters:
        stock_names (list[str]): List of ticker symbols.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping stock symbols to their OHLC data.
    """
    result = {}

    for stock_name in stock_names:
        try:
            # Create Ticker object to get stock info
            ticker = yf.Ticker(stock_name)
            
            # Get the stock's available history range
            history = ticker.history(period="max")
            if not history.empty:
                available_start = history.index[0].strftime('%Y-%m-%d')
                available_end = history.index[-1].strftime('%Y-%m-%d')
                
                # Use available dates if requested dates are out of range
                actual_start = max(start_date, available_start)
                actual_end = min(end_date, available_end)
                
                # Get the OHLC data
                ohlc_data = get_stock_ohlc(stock_name, actual_start, actual_end)
                result[stock_name] = ohlc_data
            else:
                print(f"No data available for {stock_name}")
                result[stock_name] = pd.DataFrame()
                
        except Exception as e:
            print(f"Error processing {stock_name}: {e}")
            result[stock_name] = pd.DataFrame()

    return result