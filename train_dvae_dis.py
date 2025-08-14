import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import numpy as np
import os
from tensorboardX import SummaryWriter
import pickle
import argparse
import glob 
import json

from networks.dymeshvae import DyMeshVAE
from utils.mesh_utils import get_adjacency_matrix

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

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
        return vertices, faces, valid_length, valid_mask

class DyMeshDataset_val(Dataset):
    def __init__(self, data_dir, num_t=16, max_length=4096):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])
        self.num_t = num_t
        self.num_data = len(self.files)
        self.max_length = max_length
        self.faces_max_length = int(self.max_length * 2.5)
    def __len__(self):
        return min(self.num_data, 1024) if self.num_data > 0 else 0
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
        return vertices, faces, valid_length, valid_mask

def train_epoch(model, train_loader, val_loader, optimizer, lr_scheduler, device, epoch, writer, global_iter, opt, is_ddp):
    model.train()
    is_main_process = not is_ddp or dist.get_rank() == 0
    pbar = train_loader
    if is_main_process:
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.train_epoch}", dynamic_ncols=True, initial=global_iter % len(train_loader))
    for b, bc in enumerate(pbar):
        batch, faces, valid_length, valid_mask = bc
        batch, faces, valid_length, valid_mask = batch.to(device), faces.to(device), valid_length.to(device), valid_mask.to(device)
        adj_matrix = get_adjacency_matrix(batch[:, 0], faces, valid_length)
        optimizer.zero_grad()
        output = model(batch, batch[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix)
        if opt.avg_loss:
            loss, loss_recon, loss_kl_t = loss_function_avg(batch, output, valid_mask=valid_mask)
        else:
            loss, loss_recon, loss_kl_t = loss_function(batch, output, valid_mask=valid_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        if is_main_process:
            loss_value = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=loss_value, lr=current_lr)
            if writer:
                writer.add_scalar('train/loss', loss.item(), global_iter)
                writer.add_scalar('train/loss_recon', loss_recon.item(), global_iter)
                writer.add_scalar('train/loss_kl_t', loss_kl_t.item(), global_iter)
                writer.add_scalar('train/learning_rate', current_lr, global_iter)
        if opt.validate and global_iter > 0 and global_iter % opt.validation_inter == 0:
            validate(model, val_loader, device, writer, global_iter, is_ddp, avg_loss=opt.avg_loss)
        global_iter += 1
    return global_iter

@torch.no_grad()
def validate(model, val_loader, device, writer, global_iter, is_ddp, avg_loss=False):
    model.eval()
    local_loss_sum = torch.tensor(0.0, device=device)
    local_iters_count = torch.tensor(0, device=device, dtype=torch.long)
    for b, bc in enumerate(val_loader):
        batch, faces, valid_length, valid_mask = bc
        batch, faces, valid_length, valid_mask = batch.to(device), faces.to(device), valid_length.to(device), valid_mask.to(device)
        adj_matrix = get_adjacency_matrix(batch[:, 0], faces, valid_length) 
        output = model(batch, batch[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix)
        if avg_loss:
            loss, loss_recon, loss_kl_t = loss_function_avg(batch, output, valid_mask=valid_mask)
        else:
            loss, loss_recon, loss_kl_t = loss_function(batch, output, valid_mask=valid_mask)
        local_loss_sum += loss_recon
        local_iters_count += 1
    if is_ddp:
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_iters_count, op=dist.ReduceOp.SUM)
    avg_batch_loss = local_loss_sum / local_iters_count if local_iters_count > 0 else 0
    is_main_process = not is_ddp or dist.get_rank() == 0
    if is_main_process:
        print(f'\nValidation at step {global_iter} - Average Batch Error: {avg_batch_loss.item():.6f}')
        if writer:
            writer.add_scalar('val/error', avg_batch_loss.item(), global_iter)
            writer.flush()
    model.train()

def loss_function(x, output, valid_mask=None, beta=1e-6):
    recon_x, kl_temp = output['recon_pc'], output['kl_temp']
    if valid_mask is not None:
        point_wise_dist = torch.norm(recon_x - x, p=2, dim=-1)
        masked_dist = point_wise_dist * valid_mask.unsqueeze(1)
        total_dist_sum = masked_dist.sum()
        total_valid_obs = valid_mask.sum() * x.shape[1]
        loss_rec = total_dist_sum / (total_valid_obs + 1e-8)
    else:
        loss_rec = torch.norm(recon_x - x, p=2, dim=-1).mean()
    KLD_temp = kl_temp.mean() 
    loss = loss_rec + beta * KLD_temp
    return loss, loss_rec, KLD_temp

def loss_function_avg(x, output, valid_mask=None, beta=1e-6):
    recon_x, kl_temp = output['recon_pc'], output['kl_temp']
    if valid_mask is not None:
        point_wise_dist = torch.norm(recon_x - x, p=2, dim=-1)
        masked_dist = point_wise_dist * valid_mask.unsqueeze(1)
        total_dist_sum = masked_dist.sum()
        total_valid_obs = valid_mask.sum() * x.shape[1]
        loss_rec = total_dist_sum / (total_valid_obs + 1e-8)
    else:
        loss_rec = torch.norm(recon_x - x, p=2, dim=-1).mean()
    KLD_temp = kl_temp.mean() 
    loss = loss_rec + beta * KLD_temp
    return loss, loss_rec, KLD_temp

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpts_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--validation_inter", type=int, default=400, help="Validate every N steps.")
    parser.add_argument("--save_inter", type=int, default=1, help="Save every N epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size PER GPU.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial/Max learning rate.")
    parser.add_argument("--train_epoch", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of steps for learning rate warmup.")
    parser.add_argument("--video_save_dir", type=str, default="./test_gen_videos/val")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--enc_depth", type=int, default=8)
    parser.add_argument("--dec_depth", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_t", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_traj", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--norm_type", default="qkv", choices=["q", "qk", "qkv"])
    parser.add_argument("--avg_loss", action="store_true")
    opt = parser.parse_args()

    # DDP Setup
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    
    local_rank = 0
    if is_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = not is_ddp or dist.get_rank() == 0
    writer = None
    exp_dir = os.path.join(opt.ckpts_dir, opt.exp)
    
    # DyMeshVAE config
    model_config = {
        'enc_depth': opt.enc_depth,
        'dec_depth': opt.dec_depth,
        'dim': opt.dim,
        'output_dim': 3 * opt.num_t,  
        'latent_dim': opt.latent_dim,
        'T': opt.num_t,
        'num_traj': opt.num_traj,
        'n_layers': opt.n_layers,
        'norm_type': opt.norm_type
    }

    if is_main_process:
        seed_everything(opt.seed)
        print(f"Running with options: {opt}")
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(opt.log_dir, opt.exp)
        # If resuming, do not overwrite logs
        writer = SummaryWriter(log_dir=str(log_dir), purge_step=None if opt.resume else 0)
        # save config
        config_save_path = os.path.join(exp_dir, 'model_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"Model configuration saved to {config_save_path}")

    if is_ddp:
        dist.barrier()

    # Dataset and DataLoader setup
    dataset = DyMeshDataset(opt.data_dir, num_t=opt.num_t, max_length=opt.max_length)
    train_sampler = DistributedSampler(dataset) if is_ddp else None
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=8, persistent_workers=True, pin_memory=True, drop_last=True)
    val_loader = None
    if opt.validate and opt.val_data_dir:
        val_dataset = DyMeshDataset_val(opt.val_data_dir, num_t=opt.num_t, max_length=opt.max_length)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, sampler=val_sampler,
                                num_workers=8, persistent_workers=True, pin_memory=True, drop_last=False)

    # Model setup
    model = DyMeshVAE(**model_config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    # LR Scheduler setup
    steps_per_epoch = len(train_loader)
    total_training_steps = opt.train_epoch * steps_per_epoch
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=opt.warmup_steps)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps - opt.warmup_steps, eta_min=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[opt.warmup_steps])

    # Start training
    start_epoch = 0
    global_iter = 0
    
    # Process resuming
    if opt.resume:
        # Find the latest checkpoint
        latest_ckpt_path = os.path.join(exp_dir, 'latest.pth')
        if os.path.exists(latest_ckpt_path):
            if is_main_process:
                print(f"Resuming training from checkpoint: {latest_ckpt_path}")
            # Load checkpoint on the correct device
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if is_ddp else device
            checkpoint = torch.load(latest_ckpt_path, map_location=map_location)
            # Load model state
            model_state = checkpoint['model_state_dict']
            if is_ddp and not isinstance(model, DistributedDataParallel):
                 # If current model is not DDP but checkpoint was, strip 'module.' prefix
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            model.load_state_dict(model_state)
            # Load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Load progress
            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            global_iter = checkpoint['global_iter']
            if is_main_process:
                print(f"Resumed from epoch {start_epoch}, global step {global_iter}.")
        else:
            if is_main_process:
                print("Resume flag was set, but no 'latest.pth' checkpoint found. Starting from scratch.")

    # Wrap model with DDP *after* loading state dict
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # Training loop
    for epoch in range(start_epoch, opt.train_epoch):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        global_iter = train_epoch(model, train_loader, val_loader, optimizer, lr_scheduler, device, epoch, writer, global_iter, opt, is_ddp)
        if is_main_process:
            # Create a dictionary to save all necessary states
            checkpoint = {
                'epoch': epoch,
                'global_iter': global_iter,
                'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'opt': opt # Optional: save the config as well
            }
            
            # Save a checkpoint for this specific epoch if save_inter matches
            if (epoch + 1) % opt.save_inter == 0:
                epoch_save_path = os.path.join(exp_dir, f'dvae_{epoch+1}.pth')
                torch.save(checkpoint, epoch_save_path)

            # Always save a 'latest.pth' for easy resuming
            latest_save_path = os.path.join(exp_dir, 'latest.pth')
            torch.save(checkpoint, latest_save_path)
            if (epoch + 1) % opt.save_inter == 0:
                 print(f"Saved checkpoint for epoch {epoch+1} and updated 'latest.pth'")

    # Final save
    if is_main_process:
        # Final model can just be the state dict for inference
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), os.path.join(exp_dir, 'dvae_f.pth'))
        if writer:
            writer.close()

    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
    