from utils import load_config, set_seed
from train_eval import run_experiment_with_news
from visualization import visualize_test_predictions
import torch

def main():
    # Загружаем конфиг
    config = load_config("config.yaml")
    set_seed(42)

    ticker       = config["ticker"]
    start_date   = config["start_date"]
    end_date     = config["end_date"]
    seq_length   = config["seq_length"]
    hidden_dim   = config["hidden_dim"]
    num_layers   = config["num_layers"]
    dropout      = config["dropout"]
    lr           = config["lr"]
    epochs       = config["epochs"]
    batch_size   = config["batch_size"]

    # Обучаем модель и получаем результат
    model, test_mse, (X_test, y_test, test_dates, scalers, feature_cols, df) = run_experiment_with_news(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        news_query=ticker
    )

    print(f"Test MSE: {test_mse:.6f}")
    visualize_test_predictions(model, X_test, y_test, test_dates, scalers, ticker)

if __name__ == "__main__":
    main()
