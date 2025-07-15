import pandas as pd
import ta

def SMA(series: pd.Series, window: int):   # 単純移動平均
    return series.rolling(window).mean()

def EMA(series: pd.Series, window: int):   # 指数移動平均
    return series.ewm(span=window, adjust=False).mean()

def RSI(series: pd.Series, window: int):   # RSI
    return ta.momentum.rsi(series, window)
