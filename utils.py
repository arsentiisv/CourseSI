import yaml
import random
import numpy as np
import torch

def load_config(config_file='config.yaml'):
    """
    Загружает конфигурацию из YAML-файла.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed=42):
    """
    Устанавливает фиксированный seed для воспроизводимости экспериментов.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
