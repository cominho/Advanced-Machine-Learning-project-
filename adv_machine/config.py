import pandas as pd 

import adv_machine.utils as ut 

df = pd.read_csv(f'available_stock.csv',delimiter=';')
df.index = df['Ticker']
ticker_to_screen = df.to_dict(orient='index')

available_stocks = list(ticker_to_screen.keys())