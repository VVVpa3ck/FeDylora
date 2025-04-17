import os
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# 定义保存路径
CACHE_DIR = "./cache"
MODEL_DIR = os.path.join(CACHE_DIR, "models")
DATA_DIR = os.path.join(CACHE_DIR, "datasets")

# 创建目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def download_model():
    print("Downloading BERT model and tokenizer...")
    # 下载并保存模型和 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=MODEL_DIR)
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir=MODEL_DIR)
    
    # 保存到本地
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

def download_dataset():
    print("Downloading IMDb dataset...")
    # 下载 IMDb 数据集
    dataset = load_dataset("imdb", cache_dir=DATA_DIR)
    
    # 保存到本地
    dataset.save_to_disk(DATA_DIR)
    print(f"Dataset saved to {DATA_DIR}")

if __name__ == "__main__":
    download_model()
    download_dataset()