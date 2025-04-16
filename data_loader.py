import yfinance as yf
import pandas as pd
import sys


def load_data(ticker, start_date, end_date):
    """
    Загружает исторические данные по котировкам с использованием Yahoo Finance (через yfinance).

    Аргументы:
        ticker (str): Тикер акции (например, "AAPL" или "SBER.ME").
        start_date (str): Начало периода в формате YYYY-MM-DD.
        end_date (str): Конец периода в формате YYYY-MM-DD.

    Возвращает:
        DataFrame с колонками: Open, High, Low, Close, Volume.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки данных для {ticker}: {e}")
        sys.exit(1)
    if data.empty:
        print(f"[ERROR] Нет данных для {ticker}. Проверьте соединение или входные параметры.")
        sys.exit(1)
    # Оставляем только нужные колонки
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2019-01-01"
    end_date = "2023-01-01"
    df = load_data(ticker, start_date, end_date)
    print(df.head())
