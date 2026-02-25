import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import numpy as np
import os
import argparse
import pickle
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json

from diffusion.configs import get_model_configs
from networks.configs import model_from_config
from networks.dymeshvae import DyMeshVAE
from diffusion.rf_diffusion import rf_training_losses, rf_sample
from utils.mesh_utils import get_adjacency_matrix

def setup_ddp():
    """Initializes the DDP process group using environment variables."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def seed_everything(seed):
    """Sets the random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        vertices = torch.tensor(vertices, dtype=torch.float32)[:self.num_t]
        faces = torch.tensor(faces, dtype=torch.int64)
        center = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
        vertices = vertices - center
        v_max = vertices[0].abs().max() + 1e-8
        vertices = vertices / v_max
        valid_length = vertices.shape[1]
        valid_mask = torch.ones(self.max_length, dtype=torch.bool)
        valid_mask[valid_length:] = False
        padded_vertices = torch.zeros(self.num_t, self.max_length, 3, dtype=torch.float32)
        padded_vertices[:, :valid_length] = vertices
        padded_faces = -torch.ones(self.faces_max_length, 3, dtype=torch.int64)
        padded_faces[:len(faces)] = faces
        caption = mesh_file.get("caption", " ")
        return {
            'vertices': padded_vertices, 
            'faces': padded_faces, 
            'valid_length': valid_length, 
            'valid_mask': valid_mask, 
            'caption': caption  
        }

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Distributed training script for Diffusion model on dynamic meshes.")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name, used for saving checkpoints and logs.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in the experiment directory.")
    parser.add_argument("--finetune_from", default=None, help="Finetuning from the latest checkpoint in the experiment directory.")
    # Data & Path
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dvae_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--save_dir", type=str, default="./rf_ckpts")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--json_dir", type=str, default="./dvae_factors")
    parser.add_argument("--max_length", type=int, default=4096)
    # VAE
    parser.add_argument("--vae_exp", type=str, default="dvae_v1_lr4_avg")
    parser.add_argument("--vae_epoch", type=str, default="2000")
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size PER GPU.")
    parser.add_argument("--train_epoch", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--rescale", action="store_true")
    # Validation & Saving
    # parser.add_argument("--no_val", action="store_true")
    # parser.add_argument("--validation_inter", type=int, default=2000)
    parser.add_argument("--save_inter", type=int, default=1, help="Save checkpoint every N epochs.")
    # Model
    parser.add_argument("--base_name", type=str, default="40m", choices=["40m", "300m", "1b"])
    
    opt = parser.parse_args()

    # DDP Setup
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = setup_ddp() if is_ddp else 0
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = not is_ddp or dist.get_rank() == 0

    # Paths and Logging Setup
    exp_dir = os.path.join(opt.save_dir, opt.exp)

    # --- Load VAE and Stats ---
    vae_config_dir = os.path.join(opt.dvae_dir, opt.vae_exp, "model_config.json")
    with open(vae_config_dir, 'r') as f:
        vae_model_config = json.load(f)
        opt.f0_channels = vae_model_config["latent_dim"]
        opt.input_channels = 2 * vae_model_config["latent_dim"]
        opt.num_t = vae_model_config["T"]
        opt.num_traj = vae_model_config["num_traj"]
    
    # --- Load RF configs ---
    rf_model_config = get_model_configs(opt)
    
    # all configs
    full_training_config = {
        "vae_config": vae_model_config,
        "rf_config": rf_model_config,
        "training_args": {
            "exp_name": opt.exp,
            "vae_exp_dependency": opt.vae_exp,
            "vae_epoch_dependency": opt.vae_epoch,
            "learning_rate": opt.lr,
            "batch_size_per_gpu": opt.batch_size,
            "total_epochs": opt.train_epoch,
            "seed": opt.seed,
            "rescale_stats": opt.rescale,
        }
    }

    writer = None
    if is_main_process:
        seed_everything(opt.seed)
        print(f"Starting experiment: {opt.exp}")
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(opt.log_dir, opt.exp)
        writer = SummaryWriter(log_dir=str(log_dir), purge_step=None if opt.resume else 0)
        # 保存这个总的配置文件
        config_save_path = os.path.join(exp_dir, 'training_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(full_training_config, f, indent=4)
        print(f"Full training configuration saved to {config_save_path}")

    if is_ddp:
        dist.barrier() # Ensure all processes have set up paths before proceeding
    
    # Load VAE
    vae_model = DyMeshVAE(**vae_model_config).to(device)
    vae_ckpt_path = os.path.join(opt.dvae_dir, opt.vae_exp, f"dvae_{opt.vae_epoch}.pth")
    vae_model = load_compatible_checkpoint(vae_model, vae_ckpt_path, device)
    vae_model.eval()

    # Scale
    if opt.rescale:
        json_path = os.path.join(opt.json_dir, "{}_{}.json".format(opt.vae_exp, opt.vae_epoch))
        with open(json_path, 'r') as f:
            stats = json.load(f)
        x0_mean = stats['f0_mean']
        x0_std = stats['f0_std']
        xt_mean = stats['ft_mean']
        xt_std = stats['ft_std']

    # Model, Optimizer
    rf_model = model_from_config(rf_model_config, device)
    optimizer = optim.AdamW(rf_model.parameters(), lr=opt.lr)
    
    # DataLoader Setup 
    dataset = DyMeshDataset(opt.data_dir, num_t=opt.num_t, max_length=opt.max_length)
    sampler = DistributedSampler(dataset) if is_ddp else None
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler, shuffle=(sampler is None),
                              num_workers=8, pin_memory=True, drop_last=True)
    steps_per_epoch = len(dataloader)
    total_steps = opt.train_epoch * steps_per_epoch
    
    # Setup LR scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=opt.warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - opt.warmup_steps, eta_min=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[opt.warmup_steps])
    
    # Either resume or finetune 
    if opt.resume and opt.finetune_from is not None:
        raise ValueError("Cannot use --resume and --finetune_from simultaneously.")
        
    # Resume logic
    start_epoch, global_iter = 0, 0
    if opt.resume:
        ckpt_path = os.path.join(exp_dir, 'latest.pth')
        if os.path.exists(ckpt_path):
            if is_main_process: print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            rf_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_iter = checkpoint['global_iter']
        else:
            if is_main_process: print("Resume flag set, but 'latest.pth' not found. Starting from scratch.")
    
    # Finetune logic
    if opt.finetune_from is not None:
        ckpt_path = os.path.join(opt.save_dir, opt.finetune_from, 'latest.pth')
        if os.path.exists(ckpt_path):
            if is_main_process: print(f"Finetuning from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            rf_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Finetuning from model: {ckpt_path}")
        else:
            if is_main_process: print("Finetuning flag set, but 'latest.pth' not found. Starting from scratch.")
    
    # DDP Wrapping (must be done after loading weights) 
    if is_ddp:
        rf_model = DistributedDataParallel(rf_model, device_ids=[local_rank], find_unused_parameters=True)

    # Main Training Loop 
    for epoch in range(start_epoch, opt.train_epoch):
        if is_ddp:
            sampler.set_epoch(epoch)
        pbar = dataloader
        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.train_epoch}")
        for i, data in enumerate(pbar):
            prompt = data['caption']
            vertices = data['vertices'].to(device)
            faces = data['faces'].to(device)
            valid_length = data['valid_length'].to(device)
            valid_mask = data['valid_mask'].to(device)
            adj_matrix = get_adjacency_matrix(vertices[:, 0], faces, valid_length)
            
            with torch.no_grad():
                # Encode with VAE
                x_start = vae_model(vertices, vertices[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix, num_traj=opt.num_traj, just_encode=True)
                if opt.rescale:
                    x0_start = (x_start[:, :, :opt.f0_channels] - x0_mean) / x0_std
                    xt_start = (x_start[:, :, opt.f0_channels:] - xt_mean) / xt_std
                    x_start = torch.cat([x0_start, xt_start], dim=-1)

            # Training Step 
            optimizer.zero_grad()
            model_kwargs = dict(texts=prompt)
            loss_dict = rf_training_losses(rf_model, x_start, model_kwargs=model_kwargs, f0_channels=opt.f0_channels)
            loss = loss_dict['loss'].mean()

            # Skip step on NaN/Inf
            if not torch.isfinite(loss):
                if is_main_process: print(f"Warning: NaN/Inf loss at step {global_iter}. Skipping update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(rf_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            # Logging and Validation
            if is_main_process:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=f"{lr:.2e}")
                writer.add_scalar('train/loss', loss.item(), global_iter)
                writer.add_scalar('train/lr', lr, global_iter)

                # if not opt.no_val and global_iter > 0 and global_iter % opt.validation_inter == 0:
                #     pass
            
            global_iter += 1

        # Save Checkpoint 
        if is_main_process:
            checkpoint = {
                'epoch': epoch,
                'global_iter': global_iter,
                'model_state_dict': rf_model.module.state_dict() if is_ddp else rf_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }
            if (epoch + 1) % opt.save_inter == 0:
                torch.save(checkpoint, os.path.join(exp_dir, f'rf_epoch_{epoch+1}.pth'))
            torch.save(checkpoint, os.path.join(exp_dir, 'latest.pth'))
            print(f"Epoch {epoch+1} finished. Checkpoint saved.")

    # Final Cleanup 
    if is_main_process:
        print("Training finished.")
        writer.close()
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
