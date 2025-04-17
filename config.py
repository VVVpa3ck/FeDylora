import random
import torch
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    SEED = 42
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 256
    BATCH_SIZE = 16
    NUM_CLIENTS = 10
    NUM_ROUNDS = 3
    LOCAL_EPOCHS = 1
    LR = 2e-5
    R_VALUES = [2, 4, 8, 16]
    LORA_ALPHA = 16
    DATASET = "imdb"
    NON_IID = True
    SERVER_ADDRESS = "localhost:8000"

    # 本地缓存路径
    CACHE_DIR = "./cache"
    MODEL_DIR = os.path.join(CACHE_DIR, "models")
    DATA_DIR = os.path.join(CACHE_DIR, "datasets")

    # 检查本地文件是否存在
    if not os.path.exists(MODEL_DIR) or not os.path.exists(DATA_DIR):
        raise FileNotFoundError("Please run `download_assets.py` to download model and dataset first.")

set_seed(Config.SEED)