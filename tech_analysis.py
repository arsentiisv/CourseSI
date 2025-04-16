import numpy as np
import pandas as pd
import talib


def tech_analysis(df):
    """
    Вычисляет технические индикаторы и добавляет их в DataFrame.

    Добавляются:
    - RSI
    - MACD (и его гистограмма)
    - Bollinger Bands (BB_upper, BB_middle, BB_lower)
    - ADX
    - Stochastic Oscillator (%K и %D)

    Если DataFrame имеет MultiIndex по столбцам, приводим к одноуровневому.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    open_ = df['Open'].values
    high_ = df['High'].values
    low_ = df['Low'].values
    close_ = df['Close'].values
    volume_ = df['Volume'].values

    # RSI
    df['RSI'] = talib.RSI(close_, timeperiod=14)

    # MACD (возвращает tuple: macd, signal, hist)
    df['MACD'], df['Signal'], df['MACD_Hist'] = talib.MACD(close_, fastperiod=12, slowperiod=26, signalperiod=9)

    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close_, timeperiod=20, nbdevup=2, nbdevdn=2,
                                                                   matype=0)

    # ADX
    df['ADX'] = talib.ADX(high_, low_, close_, timeperiod=14)

    # Stochastic Oscillator (%K и %D)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high_, low_, close_,
                                               fastk_period=14, slowk_period=3,
                                               slowk_matype=0, slowd_period=3, slowd_matype=0)

    return df
