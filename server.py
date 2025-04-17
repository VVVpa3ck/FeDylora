import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_utils import DyLoRAModel, get_parameters, set_parameters, truncate_or_extend, compute_importance
from config import Config
from data_utils import load_data
import numpy as np

def evaluate_global_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(Config.DEVICE), attention_mask.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total, total_loss / len(dataloader)

class DRAStrategy(fl.server.strategy.Strategy):
    def __init__(self, model, test_dataloader, pretrain_model):
        self.model = model
        self.test_dataloader = test_dataloader
        self.pretrain_model = pretrain_model
        self.r_g = None

    def initialize_parameters(self, client_manager):
        return get_parameters(self.model)

    def configure_fit(self, server_round, parameters, client_manager):
        # 模拟中直接返回所有客户端，无需复杂配置
        return [(client, fl.common.FitIns(parameters, {})) for client in client_manager.clients.values()]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        
        client_params = [r.parameters for _, r in results]
        client_ranks = [r.metrics["r"] for _, r in results]
        client_importance = [r.metrics["importance"] for _, r in results]
        weights = [r.num_examples for _, r in results]
        total_weight = sum(weights)

        self.r_g = int(np.average(client_ranks, weights=weights))
        print(f"Round {server_round} - Global rank r_g: {self.r_g}")

        aggregated_params = []
        param_keys = self.model.state_dict().keys()
        for idx, key in enumerate(param_keys):
            if "lora_A" in key or "lora_B" in key:
                adjusted_params = []
                for i, client_param in enumerate(client_params):
                    param = torch.from_numpy(client_param[idx]).to(Config.DEVICE)
                    importance = torch.from_numpy(client_importance[i][key]).to(Config.DEVICE)
                    pretrain_param = self.pretrain_model.state_dict()[key]
                    adjusted_param = truncate_or_extend(param, self.r_g, importance, pretrain_param)
                    adjusted_params.append(adjusted_param.cpu().numpy())
                aggregated_param = np.average(adjusted_params, axis=0, weights=weights)
                aggregated_params.append(aggregated_param)
            else:
                param_stack = [c[idx] for c in client_params]
                aggregated_param = np.average(param_stack, axis=0, weights=weights)
                aggregated_params.append(aggregated_param)

        set_parameters(self.model, aggregated_params, self.pretrain_model)
        self.model.set_rank(self.r_g)
        
        accuracy, loss = evaluate_global_model(self.model, self.test_dataloader)
        print(f"Round {server_round} - Global Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        return aggregated_params, {"accuracy": accuracy, "r_g": self.r_g}

    def configure_evaluate(self, server_round, parameters, client_manager):
        # 模拟中无需客户端评估，直接返回空列表
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        # 模拟中无需聚合客户端评估结果，返回空字典
        return None, {}

    def evaluate(self, server_round, parameters):
        # 在服务器端直接评估全局模型
        set_parameters(self.model, parameters, self.pretrain_model)
        accuracy, loss = evaluate_global_model(self.model, self.test_dataloader)
        return loss, {"accuracy": accuracy}

def main():
    client_datasets, test_dataset, _ = load_data()
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    model = DyLoRAModel(r=max(Config.R_VALUES)).to(Config.DEVICE)
    pretrain_model = DyLoRAModel(r=max(Config.R_VALUES)).to(Config.DEVICE)
    
    strategy = DRAStrategy(model, test_dataloader, pretrain_model)
    fl.server.start_server(
        server_address=Config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=Config.NUM_ROUNDS),
        strategy=strategy
    )

if __name__ == "__main__":
    main()