import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from data_loader import load_data  # используем загрузчик с Yahoo
from tech_analysis import tech_analysis
from models import StockLSTM


def normalize_data(df, ticker, feature_cols):
    scalers = {}
    normalized_df = df.copy()
    for col in feature_cols:
        scaler = MinMaxScaler()
        normalized_df[col] = scaler.fit_transform(df[[col]])
        scalers[(col, ticker)] = scaler
    return normalized_df, scalers


def create_sequences_with_dates(data_values, data_dates, seq_length):
    X, y, label_dates = [], [], []
    for i in range(len(data_values) - seq_length):
        X.append(data_values[i:i + seq_length])
        y.append(data_values[i + seq_length, 0])  # прогнозируем 'Close'
        label_dates.append(data_dates[i + seq_length])
    return np.array(X), np.array(y), np.array(label_dates)


def split_data(X, y, dates, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:], dates[:train_size], dates[train_size:]


def prepare_dataloader(X_train, y_train, X_test, y_test, batch_size=32):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).squeeze()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).squeeze()
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, optimizer, criterion, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}")


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb).squeeze()
            loss = criterion(preds, yb)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def get_scaler(scalers, column, ticker):
    key = (column, ticker)
    if key in scalers:
        return scalers[key]
    for k in scalers:
        if k[0] == column and k[1] == ticker:
            return scalers[k]
    raise KeyError(f"Scaler for {column} not найден. Available keys: {list(scalers.keys())}")


def run_experiment_with_news(ticker, start_date, end_date, seq_length, hidden_dim, num_layers, dropout, lr, epochs,
                             batch_size, news_query):
    """
    Загружает данные с Yahoo, применяет технический анализ,
    нормализует данные, формирует последовательности для LSTM, обучает модель и возвращает модель и MSE.
    Интеграция новостного сентимента отключена (так как мы работаем только с Yahoo).
    """
    df = load_data(ticker, start_date, end_date)
    df = tech_analysis(df)
    df.dropna(inplace=True)

    feature_cols = ['Close', 'Volume', 'RSI', 'MACD_Hist', 'ADX',
                    'BB_upper', 'BB_middle', 'BB_lower', 'STOCH_K', 'STOCH_D']

    if 'sentiment' in df.columns:
        feature_cols.append('sentiment')

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Столбец {col} не найден в DataFrame")

    normalized_df, scalers = normalize_data(df, ticker, feature_cols)
    data_dates = normalized_df.index
    data_values = normalized_df[feature_cols].values

    X, y, label_dates = create_sequences_with_dates(data_values, data_dates, seq_length)
    X_train, X_test, y_train, y_test, dates_train, dates_test = split_data(X, y, label_dates)
    train_loader, test_loader = prepare_dataloader(X_train, y_train, X_test, y_test, batch_size)

    input_dim = len(feature_cols)
    output_dim = 1
    model = StockLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_model(model, train_loader, optimizer, criterion, num_epochs=epochs)
    test_mse = evaluate_model(model, test_loader, criterion)
    print(f"Test MSE = {test_mse:.6f}")

    return model, test_mse, (X_test, y_test, dates_test, scalers, feature_cols, df)
