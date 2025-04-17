# client.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model_utils import DyLoRAModel, get_parameters, set_parameters, compute_importance, truncate_or_extend
from config import Config
import random

def compute_communication_cost(parameters):
    return sum(p.nbytes for p in parameters)

def train(model, dataloader, epochs=Config.LOCAL_EPOCHS):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(Config.DEVICE), attention_mask.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(Config.DEVICE), attention_mask.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

class Client:
    def __init__(self, cid, dataset):
        self.cid = cid
        self.dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        self.r = random.choice(Config.R_VALUES)
        self.model = DyLoRAModel(r=self.r).to(Config.DEVICE)
        self.model.set_rank(self.r)
        print(f"Client {cid} initialized with rank {self.r}, dataset size: {len(dataset)}")

    def fit(self, parameters):
        set_parameters(self.model, parameters)
        train(self.model, self.dataloader)
        
        updated_params = []
        importance_dict = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                importance = compute_importance(param, param.grad)
                importance_dict[name] = importance.cpu().detach().numpy()  # 确保 importance 也分离梯度
            updated_params.append(param.cpu().detach().numpy())  # 使用 detach() 分离梯度
        
        comm_cost = compute_communication_cost(updated_params)
        print(f"Client {self.cid} - Communication cost: {comm_cost} bytes")
        
        return updated_params, len(self.dataloader.dataset), {
            "comm_cost": comm_cost,
            "r": self.r,
            "importance": importance_dict
        }

    def evaluate(self, parameters):
        set_parameters(self.model, parameters)
        accuracy = evaluate(self.model, self.dataloader)
        return accuracy, len(self.dataloader.dataset), {"accuracy": accuracy}

def simulate_clients(client_datasets):
    clients = [Client(cid, dataset) for cid, dataset in enumerate(client_datasets)]
    return clients