#models.py
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        LSTM-модель для прогнозирования котировок.

        :param input_dim: Количество входных признаков.
        :param hidden_dim: Размер скрытого слоя.
        :param num_layers: Количество LSTM-слоёв.
        :param output_dim: Размер выходного слоя (например, 1 для прогноза цены).
        :param dropout: Доля dropout между LSTM-слоями.
        """
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
