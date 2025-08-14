import torch
from .transformer import DyMeshMMDiT

def model_from_config(config, device):
    config = config.copy()
    return DyMeshMMDiT(device=device, dtype=torch.float32, **config)
    