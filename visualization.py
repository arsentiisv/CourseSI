import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch

def get_scaler(scalers, column, ticker):
    key_direct = (column, ticker)
    if key_direct in scalers:
        return scalers[key_direct]
    for k in scalers.keys():
        if isinstance(k, tuple) and len(k) == 2:
            if k[0] == column and k[1] == ticker:
                return scalers[k]
    raise KeyError(f"Scaler for {column} not найден. Available keys: {list(scalers.keys())}")

def visualize_test_predictions(model, X_test, y_test, test_dates, scalers, ticker):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

    close_scaler = get_scaler(scalers, 'Close', ticker)
    preds_unscaled = close_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_test_unscaled = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test_unscaled,
        mode='lines',
        name=f"Real {ticker}",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=preds_unscaled,
        mode='lines',
        name=f"Predicted {ticker}",
        line=dict(color='red')
    ))
    fig.update_layout(
        title=f"Stock Price Prediction for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    fig.show()
