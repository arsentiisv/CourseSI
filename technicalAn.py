import numpy as np
import pandas as pd
import pandas_ta as ta

class Technic:
    def tech(self, td):
        ### Должна поступить таблица с [date, Volume, High, Open, Low, Close]
        ### RSI
        td['RSI'] = ta.rsi(td['Close'])

        ### EMA12, EMA26, EMA5, EMA6,EMA10, EMA20, EMA50, EMA100, EMA200
        for per in [5, 6, 10, 12, 20, 26, 50, 100, 200]:
            td[f'EMA{per}'] = td['Close'].ewm(span=per, adjust=False).mean()

        ### MACD, SignalMACD, MACDHIST
        td['MACD'] = td['EMA12'] - td['EMA26']
        td['Signal'] = td['MACD'].ewm(span=9, adjust=False).mean()
        td['MACD_H'] = td['MACD'] - td['Signal']

        ### ADL и Осциллятор Чайкина
        perS = 3
        perL = 10
        mfm = ((td['Close'] - td['Low']) - (td['High'] - td['Close'])) / (td['High'] - td['Low'])
        mfm = mfm.fillna(0)
        mfv = mfm * td['Volume']
        td['ADL'] = mfv.cumsum()
        td['EMA_S'] = td['ADL'].ewm(span=perS, adjust=False).mean()
        td['EMA_L'] = td['ADL'].ewm(span=perL, adjust=False).mean()
        td['ChO'] = td['EMA_S'] - td['EMA_L']

        ### MFI
        td['MFI'] = ta.volume.mfi(high=td['High'], low=td['Low'], close=td['Close'], volume=td['Volume'], length=14)

        ### ADX
        ted = ta.adx(high=td['High'], low=td['Low'], close=td['Close'], length=14)
        td['ADX14'] = ted['ADX_14']
        td['ADXR142'] = ted['ADXR_14_2']
        td['DMP14'] = ted['DMP_14']
        td['DMN14'] = ted['DMN_14']
        del ted

        ### Williams
        td['WILL14'] = ta.momentum.willr(td['High'], td['Low'], td['Close'], length=14)

        ### CCI
        td['CCI14'] = ta.cci(high=td['High'], low=td['Low'], close=td['Close'], length=14)

        ### Расчёт Momentum с периодом 10
        td['MOM10'] = ta.mom(td['Close'], length=10)

        ### Расчёт ROC с периодом 10
        td['ROC10'] = ta.roc(td['Close'], length=10)

        ### Coppock curve
        td['COPC'] = ta.coppock(close=td['Close'], fast=11, slow=14, length=10)

        ### Stochastic Oscillator
        stoch = ta.stoch(high=td['High'], low=td['Low'], close=td['Close'], k=14, d=3, smooth_k=3)
        td['k'] = stoch['STOCHk_14_3_3']
        td['d'] = stoch['STOCHd_14_3_3']
        del stoch

        ### Bolindger Bands
        bbands = ta.bbands(close=td['Close'], length=20, std=2)
        td['BBL'] = bbands['BBL_20_2']
        td['BBM'] = bbands['BBM_20_2']
        td['BBU'] = bbands['BBU_20_2']
        del bbands

        ### OBV
        td['OBV'] = ta.obv(close=td['Close'], volume=td['Volume'])

        ### ATR
        td['ATR'] = ta.atr(high=td['High'], low=td['Low'], close=td['Close'], length=14)

        ### Добавление лагов цены закрытия на 1 и 2 периода
        td['Close_lag1'] = td['Close'].shift(1)
        td['Close_lag2'] = td['Close'].shift(2)

        ### Скользящие статистики
        for period in [5, 10, 20]:
            td[f'SMA{period}'] = td['Close'].rolling(window=period).mean()
            td[f'STD{period}'] = td['Close'].rolling(window=period).std()

        ### Минимум и максимум цены закрытия за 5, 10 и 20 периодов
        for period in [5, 10, 20]:
            td[f'MIN{period}'] = td['Close'].rolling(window=period).min()
            td[f'MAX{period}'] = td['Close'].rolling(window=period).max()

        ### Скользящая корреляция между ценой закрытия и объёмом за 20 периодов
        td['Corr_Close_Volume'] = td['Close'].rolling(window=20).corr(td['Volume'])

        ### Размер тела свечи
        td['Body'] = (td['Close'] - td['Open']).abs()

        ### Длина верхней тени
        td['Upper_Shadow'] = td['High'] - td[['Open', 'Close']].max(axis=1)

        ### Длина нижней тени
        td['Lower_Shadow'] = td[['Open', 'Close']].min(axis=1) - td['Low']

        ### Соотношение тела свечи к общей длине
        td['Body_to_Range'] = td['Body'] / (td['High'] - td['Low'])

        ### Свечные паттерны
        patterns = [
            "morningstar",
            "engulfing",
            "hammer",
            "3whitesoldiers",
            "harami",
            "abandonedbaby",
            "eveningstar",
            "gravestonedoji",
            "3blackcrows",
            "piercing",
            "morningdojistar"
        ]
        td.ta.cdl_pattern(name=patterns, append=True)

        ### Чистка
        td = td.dropna().reset_index(drop=True)

        return td

