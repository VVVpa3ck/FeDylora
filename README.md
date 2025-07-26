# FeDyLoRA

This project demonstrates dynamic Low-Rank Adaptation (LoRA) in a federated learning
setting. Flower is used to coordinate multiple simulated clients that train
LoRA-augmented BERT models on the IMDb sentiment analysis dataset. Each client
adopts a different rank configuration while the server dynamically aggregates the
updates.

## Prerequisites
- Run `download_assets.py` before starting training. This downloads the
  pre-trained BERT model and the IMDb dataset into `./cache`.
- Python packages: `torch`, `transformers`, `datasets` and `flwr` are required.
  Install them via `pip`.
- Training expects a GPU to be available.

## Usage
1. Start the Flower server:
   ```bash
   python server.py
   ```
2. In a separate process run the main script to simulate clients:
   ```bash
   python main.py
   ```
3. Adjust ranks, number of rounds and other options in `config.py`.

## License and Citation
Parts of the code originate from Huawei Technologies and are licensed under
Apache-2.0 as noted in `DyLoRA.py`. The layer implementations
also incorporate Microsoft code released under the MIT License as referenced at
the top of `dylora_layers.py`.

