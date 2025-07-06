import yfinance as yf
import pandas as pd

# Seleccionamos 5 índices para el ejemplo
tickers = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^N225']

# ESTOS SON: S&P 500, Dow Jones, Nasdaq, FTSE 100, Nikkei 225

data = yf.download(tickers, start="2020-01-01", end="2023-01-01")
data.dropna(inplace=True)


close_data = data.xs('Close', axis=1, level=0)
#print(close_data.head())


log_returns = np.log(close_data / close_data.shift(1)).dropna()
#log_returns.plot(figsize=(12,6), title='Retornos logarítmicos')


import numpy as np

log_returns = np.log(data / data.shift(1)).dropna()

