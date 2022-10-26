from datetime import datetime, timedelta
from yahoo_fin import stock_info as si
import pandas as pd

# def get_last_stock_price(ticker, last=False):
#     if last:
#         now = datetime.now()
#         start_date = now - timedelta(days=30)
#         return si.get_data(ticker, start_date)
#     return si.get_data(ticker)

def get_last_stock_price(Ticker, last=False):
    Ticker = Ticker.replace('.', '-')
    df = si.get_data(Ticker)
    df = df.loc[df.index > pd.to_datetime(f'01-01-2020')]
    df['Diff'] = df['close'].diff()
    df['B/S'] = df['Diff'].map(lambda x : 'Sell' if x >0 else 'Buy')
    df = df.dropna()
    if last:
        now = datetime.now()
        start_date = now - timedelta(days=30)
        df = df.loc[df.index > start_date]
    return df