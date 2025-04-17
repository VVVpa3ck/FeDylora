from transformers import BertTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
from config import Config

def preprocess_data(dataset, tokenizer, max_len):
    input_ids, attention_masks, labels = [], [], []
    for example in dataset:
        text, label = example["text"], example["label"]
        encoded = tokenizer(
            text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
        )
        input_ids.append(encoded["input_ids"].squeeze(0))
        attention_masks.append(encoded["attention_mask"].squeeze(0))
        labels.append(label)
    return (torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels))

def split_iid(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    return [Subset(dataset, range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]

def split_noniid(dataset, num_clients, num_classes=2):
    indices_per_class = [[] for _ in range(num_classes)]
    for idx, (_, _, label) in enumerate(dataset):
        indices_per_class[label.item()].append(idx)
    client_data = []
    for i in range(num_clients):
        client_indices = []
        for c in range(num_classes):
            client_indices.extend(indices_per_class[c][i::num_clients])
        client_data.append(Subset(dataset, client_indices))
    return client_data

def load_data():
    # 从本地加载 tokenizer 和数据集
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_DIR, local_files_only=True)
    dataset = load_from_disk(Config.DATA_DIR)
    
    train_data = preprocess_data(dataset["train"], tokenizer, Config.MAX_LEN)
    test_data = preprocess_data(dataset["test"], tokenizer, Config.MAX_LEN)
    
    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)
    
    if Config.NON_IID:
        client_datasets = split_noniid(train_dataset, Config.NUM_CLIENTS)
    else:
        client_datasets = split_iid(train_dataset, Config.NUM_CLIENTS)
    
    return client_datasets, test_dataset, tokenizer

if __name__ == "__main__":
    client_datasets, test_dataset, _ = load_data()
    print(f"Loaded {len(client_datasets)} client datasets, test size: {len(test_dataset)}")