import flwr as fl
from data_utils import load_data
from client import simulate_clients
from server import DRAStrategy, evaluate_global_model
from model_utils import DyLoRAModel, get_parameters
from config import Config
from torch.utils.data import DataLoader
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    client_datasets, test_dataset, _ = load_data()
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    global_model = DyLoRAModel(r=max(Config.R_VALUES)).to(Config.DEVICE)
    pretrain_model = DyLoRAModel(r=max(Config.R_VALUES)).to(Config.DEVICE)
    global_params = get_parameters(global_model)
    
    clients = simulate_clients(client_datasets)
    
    for round_num in range(Config.NUM_ROUNDS):
        logging.info(f"Round {round_num + 1}/{Config.NUM_ROUNDS}")
        client_results = []
        
        for client in clients:
            params, num_examples, metrics = client.fit(global_params)
            client_results.append((None, fl.common.FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="Success"),  # 添加 status
                parameters=params,
                num_examples=num_examples,
                metrics=metrics
            )))
            torch.cuda.empty_cache()
        
        strategy = DRAStrategy(global_model, test_dataloader, pretrain_model)
        global_params, metrics = strategy.aggregate_fit(round_num + 1, client_results, [])
        logging.info(f"Global accuracy: {metrics['accuracy']:.4f}, r_g: {metrics['r_g']}")

if __name__ == "__main__":
    main()