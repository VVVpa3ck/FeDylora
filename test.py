from model_utils import DyLoRAModel
from config import Config

model = DyLoRAModel(r=8).to(Config.DEVICE)
model.set_rank(4)  # 测试 set_rank
print("Rank set successfully!")