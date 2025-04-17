import torch
import torch.nn as nn
from transformers import BertModel
from dylora_layers import Linear, Embedding
from config import Config

class DyLoRAModel(nn.Module):
    def __init__(self, r=8, lora_alpha=Config.LORA_ALPHA):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.MODEL_DIR, local_files_only=True)
        self.r = r
        
        self.bert.embeddings.word_embeddings = Embedding(
            num_embeddings=self.bert.config.vocab_size,
            embedding_dim=self.bert.config.hidden_size,
            r=r, lora_alpha=lora_alpha
        )
        for layer in self.bert.encoder.layer:
            layer.intermediate.dense = Linear(
                in_features=self.bert.config.hidden_size,
                out_features=self.bert.config.intermediate_size,
                r=r, lora_alpha=lora_alpha
            )
            layer.output.dense = Linear(
                in_features=self.bert.config.intermediate_size,
                out_features=self.bert.config.hidden_size,
                r=r, lora_alpha=lora_alpha
            )
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

    def set_rank(self, rank):
        for layer in self.bert.encoder.layer:
            if isinstance(layer.intermediate.dense, Linear):
                layer.intermediate.dense.set_rank(rank)
            if isinstance(layer.output.dense, Linear):
                layer.output.dense.set_rank(rank)

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters, pretrain_model=None):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v).to(Config.DEVICE) for k, v in params_dict}
    current_state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        current_param = current_state_dict[name]
        if "lora_A" in name and param.shape[0] != current_param.shape[0]:
            print(f"Adjusting {name}: {param.shape} -> {current_param.shape}")
            importance = compute_importance(param)
            state_dict[name] = truncate_or_extend(param, current_param.shape[0], importance, 
                                                 pretrain_model.state_dict().get(name) if pretrain_model else None)
        elif "lora_B" in name and param.shape[1] != current_param.shape[1]:
            print(f"Adjusting {name}: {param.shape} -> {current_param.shape}")
            importance = compute_importance(param.T)
            state_dict[name] = truncate_or_extend(param.T, current_param.shape[1], importance, 
                                                 pretrain_model.state_dict().get(name).T if pretrain_model else None).T
    
    model.load_state_dict(state_dict, strict=True)

def compute_importance(matrix, gradient=None):
    if gradient is None:
        return torch.norm(matrix, p=2, dim=1)
    return torch.norm(gradient, p=2, dim=1)

def truncate_or_extend(matrix, target_r, importance, pretrain_matrix=None):
    current_r = matrix.shape[0]
    if current_r == target_r:
        return matrix
    
    _, indices = torch.sort(importance, descending=True)
    
    if current_r > target_r:
        return matrix[indices[:target_r]]
    else:
        extra_r = target_r - current_r
        if pretrain_matrix is not None and pretrain_matrix.shape == matrix.shape:
            pretrain_importance = compute_importance(pretrain_matrix)
            _, pretrain_indices = torch.sort(pretrain_importance, descending=True)
            extra = pretrain_matrix[pretrain_indices[:extra_r]]
        else:
            extra = torch.zeros(extra_r, matrix.shape[1], device=matrix.device)
        return torch.cat([matrix[indices], extra], dim=0)