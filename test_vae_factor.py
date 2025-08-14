import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import os
import time
import pickle
import json
import argparse

from networks.dymeshvae import DyMeshVAE
from utils.mesh_utils import get_adjacency_matrix
from tqdm import tqdm

def setup_ddp():
    """Initializes the DDP process group using environment variables."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def load_compatible_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    else:
        model_weights = checkpoint
    if list(model_weights.keys())[0].startswith('module.'):
        model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}
    model.load_state_dict(model_weights)
    print("Model weights loaded successfully.")
    return model

class OnlineStats:
    """
    Computes mean and standard deviation in a streaming, memory-efficient manner.
    """
    def __init__(self, device='cpu'):
        self.n = torch.tensor(0, dtype=torch.long, device=device)
        self.mean = torch.tensor(0.0, dtype=torch.double, device=device)
        self.m2 = torch.tensor(0.0, dtype=torch.double, device=device)
        self.device = device

    def update(self, batch: torch.Tensor):
        batch = batch.to(self.device, dtype=torch.double)
        batch_n = batch.numel()
        if batch_n == 0:
            return
        batch_mean = torch.mean(batch)
        batch_m2 = torch.sum((batch - batch_mean) ** 2)
        new_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * batch_n / new_n
        self.m2 += batch_m2 + delta**2 * self.n * batch_n / new_n
        self.n = new_n

    @property
    def avg(self):
        return self.mean.item()

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        return torch.sqrt(self.m2 / (self.n - 1)).item()
    
    def get_state(self):
        return torch.stack([self.n.double(), self.mean, self.m2])
    
    def set_state(self, state_tensor):
        self.n = state_tensor[0].long()
        self.mean = state_tensor[1]
        self.m2 = state_tensor[2]

class DyMeshDataset(Dataset):
    def __init__(self, data_dir, num_t=16, max_length=4096):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])
        self.num_t = num_t
        self.num_data = len(self.files)
        self.max_length = max_length
        self.faces_max_length = int(self.max_length * 2.5)
    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'rb') as f:
            mesh_file = pickle.load(f)
        vertices, faces = mesh_file["vertices"], mesh_file["faces"]
        vertices, faces = torch.tensor(vertices, dtype=torch.float32)[:self.num_t], torch.tensor(faces, dtype=torch.int64)
        center = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
        vertices = vertices - center
        v_max = vertices[0].abs().max() + 1e-8
        vertices = vertices / v_max
        valid_length = vertices.shape[1]
        valid_mask = torch.cat([torch.ones(valid_length, dtype=torch.bool), torch.zeros((self.max_length-valid_length), dtype=torch.bool)], dim=0)
        vertices = torch.cat([vertices, torch.zeros(vertices.shape[0], self.max_length-vertices.shape[1], 3)], dim=1)
        faces = torch.cat([faces, -1 * torch.ones(self.faces_max_length-faces.shape[0], 3).to(torch.int64)], dim=0)
        return {'vertices': vertices, 'faces': faces, 'valid_length': valid_length, 'valid_mask': valid_mask}

def main():
    parser = argparse.ArgumentParser(description="Efficiently compute latent statistics for a dynamic mesh dataset.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--vae_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--save_dir", type=str, default="./dvae_factors")
    parser.add_argument("--vae_exp", type=str, required=True)
    parser.add_argument("--vae_epoch", type=str, default='1000')
    parser.add_argument("--enc_depth", type=int, default=8)
    parser.add_argument("--dec_depth", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_t", type=int, default=16)
    parser.add_argument("--num_traj", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--norm_type", default="qkv", choices=["q", "qk", "qkv"])
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    opt = parser.parse_args()

    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = setup_ddp() if is_ddp else 0
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = not is_ddp or dist.get_rank() == 0

    if is_main_process:
        print("Starting feature statistics calculation...")
        print(f"Running in {'DDP' if is_ddp else 'single-GPU'} mode.")
        
    # DataLoader Setup 
    dataset = DyMeshDataset(opt.data_dir, num_t=opt.num_t, max_length=opt.max_length)
    sampler = DistributedSampler(dataset, shuffle=False) if is_ddp else None
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=False)
    
    # VAE Model Loading 
    config_dir = os.path.join(opt.vae_dir, opt.vae_exp, "model_config.json")
    with open(config_dir, 'r') as f:
        model_config = json.load(f)
    opt.latent_dim = model_config["latent_dim"]
    vae_model = DyMeshVAE(**model_config).to(device)
    vae_dir = os.path.join(opt.vae_dir, opt.vae_exp, f"dvae_{opt.vae_epoch}.pth")
    vae_model = load_compatible_checkpoint(vae_model, vae_dir, device)
    vae_model.eval()
    
    # --- Online Statistics Calculation ---
    f0_stats = OnlineStats(device=device)
    ft_stats = OnlineStats(device=device)

    # Use tqdm only on the main process
    pbar = dataloader
    if is_main_process:
        pbar = tqdm(dataloader, desc="Processing batches")

    with torch.no_grad():
        for data in pbar:
            # Feature extraction
            x = data['vertices'].to(device)
            faces = data['faces'].to(device)
            valid_length = data['valid_length'].to(device)
            adj_matrix = get_adjacency_matrix(x[:, 0], faces, valid_length)
            
            x_start = vae_model(x, queries=x[:, 0], faces=faces, 
                                valid_mask=data['valid_mask'].to(device), 
                                adj_matrix=adj_matrix, 
                                just_encode=True)
            
            # Update online stats 
            f0_batch = x_start[:, :, :opt.latent_dim]
            ft_batch = x_start[:, :, opt.latent_dim:]
            f0_stats.update(f0_batch.flatten())
            ft_stats.update(ft_batch.flatten())

    # Aggregate results from all GPUs 
    if is_ddp:
        if is_main_process:
            print("Aggregating statistics across all processes...")
        
        # Combine the states [n, mean, M2] from all ranks
        f0_state = f0_stats.get_state()
        dist.all_reduce(f0_state, op=dist.ReduceOp.SUM)
        
        ft_state = ft_stats.get_state()
        dist.all_reduce(ft_state, op=dist.ReduceOp.SUM)
        
        # Only the main process needs to compute the final combined stats
        if is_main_process:
            f0_stats.set_state(f0_state)
            ft_stats.set_state(ft_state)

    # Save Final Statistics 
    if is_main_process:
        stats_dict = {
            'f0_mean': f0_stats.avg,
            'f0_std': f0_stats.std,
            'ft_mean': ft_stats.avg,
            'ft_std': ft_stats.std
        }
        
        os.makedirs(opt.save_dir, exist_ok=True)
        save_path = os.path.join(opt.save_dir, f"{opt.vae_exp}_{opt.vae_epoch}.json")
        
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)
        
        print(f"\nGlobal statistics saved to {save_path}")
        print(f"f0 mean: {f0_stats.avg:.6f}, std: {f0_stats.std:.6f}")
        print(f"ft mean: {ft_stats.avg:.6f}, std: {ft_stats.std:.6f}")

if __name__ == '__main__':
    main()
