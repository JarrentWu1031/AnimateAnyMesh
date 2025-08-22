import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import json
import numpy as np

from diffusion.rf_diffusion import rf_sample
from networks.configs import model_from_config
from networks.dymeshvae import DyMeshVAE
from utils.mesh_utils import get_adjacency_matrix, merge_identical_vertices_with_indices
from utils.render import clear_scene, import_glb, get_all_vertices, get_all_faces, drive_mesh_with_trajs_frames, drive_mesh_with_trajs_frames_five_views

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

def main(opt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Video save dir
    video_save_dir = os.path.join(opt.video_save_dir, opt.rf_exp)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    
    # Load the unified training configuration file
    print("Loading unified training configuration...")
    config_path = os.path.join(opt.rf_model_dir, opt.rf_exp, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Unified config not found at: {config_path}. This file is essential for inference.")
    
    with open(config_path, 'r') as f:
        full_config = json.load(f)

    # Extract the individual configurations
    vae_config = full_config['vae_config']
    rf_config = full_config['rf_config']
    training_args = full_config['training_args']
    opt.f0_channels = vae_config["latent_dim"]
    opt.num_t = vae_config["T"]
    opt.vae_exp = training_args["vae_exp_dependency"]
    opt.vae_epoch = training_args["vae_epoch_dependency"]
    print("Configuration loaded successfully.")
    
    # Load params
    if opt.rescale:
        json_path = os.path.join(opt.json_dir, "{}_{}.json".format(opt.vae_exp, opt.vae_epoch))
        with open(json_path, 'r') as f:
            stats = json.load(f)
        x0_mean = stats['f0_mean']
        x0_std = stats['f0_std']
        xt_mean = stats['ft_mean']
        xt_std = stats['ft_std']

    # VAE
    print("Loading DyMeshVAE...")
    vae_dir = os.path.join(opt.vae_dir, opt.vae_exp, "dvae_{}.pth".format(opt.vae_epoch))
    vae_model = DyMeshVAE(**vae_config).to(device)
    vae_model = load_compatible_checkpoint(vae_model, vae_dir, device)
    vae_model.eval()
    print("DyMeshVAE loaded!!!")
    
    # RF model
    print("Loading RF Model...")
    rf_model_dir = os.path.join(opt.rf_model_dir, opt.rf_exp, "rf_epoch_{}.pth".format(opt.rf_epoch))
    rf_model = model_from_config(rf_config, device)
    rf_model = load_compatible_checkpoint(rf_model, rf_model_dir, device)
    rf_model.eval()
    print("RF Model loaded!!!")

    with torch.no_grad():
        # load static mesh & merge vertices/faces
        filepath = os.path.join(opt.data_dir, opt.test_name+".glb")
        clear_scene()
        mesh_objects = import_glb(filepath)
        all_vertices = get_all_vertices(mesh_objects)
        all_faces = get_all_faces(mesh_objects)
        merged_verts, merged_faces, all_indices = merge_identical_vertices_with_indices(all_vertices, all_faces)
        vertices, faces = torch.tensor(merged_verts, dtype=torch.float32), torch.tensor(merged_faces, dtype=torch.int64)
        
        if opt.num_traj <= 0:
            opt.num_traj = max(512, vertices.shape[0]//8)
            print("The number of sampled trajs is not specified, set to {} by default ({} vertices in total)!!!".format(opt.num_traj, vertices.shape[0]))
        
        # recenter & rescale    
        center = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
        vertices = vertices - center
        v_max = vertices.abs().max()
        vertices = vertices / (v_max + 1e-8)

        # padding & reshape
        opt.max_length = max(opt.max_length, vertices.shape[0])
        faces_max_length = min(int(opt.max_length * 2.5), faces.shape[0])
        vertices = torch.cat([vertices, torch.zeros(opt.max_length-vertices.shape[0], 3)], dim=0)
        faces = torch.cat([faces, -1 * torch.ones(faces_max_length-faces.shape[0], 3).to(torch.int64)], dim=0)
        vertices = vertices[None, None].to(device)
        vertices = vertices.repeat(1, opt.num_t, 1, 1) 
        faces = faces[None].to(device)
        
        # prepare valid_mask, adj_matrix
        valid_mask = ~(vertices.permute(0, 2, 1, 3).flatten(2, 3) == 0.0).all(dim=-1)
        valid_length = valid_mask.sum(dim=-1)
        adj_matrix = get_adjacency_matrix(vertices[:, 0], faces, valid_length)

        # DyMeshVAE encoding
        print("Animation prompt: ", opt.prompt)
        model_kwargs = dict(texts=[opt.prompt])
        x_start = vae_model(vertices, vertices[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix, num_traj=opt.num_traj, just_encode=True)
        if opt.rescale:
            x0_start = (x_start[:, :, :opt.f0_channels] - x0_mean) / x0_std
            xt_start = (x_start[:, :, opt.f0_channels:] - xt_mean) / xt_std
            x_start = torch.cat([x0_start, xt_start], dim=-1)
        f0 = x_start[:, :, :opt.f0_channels]
        
        # RF sampling
        print("Start RF sampling...")
        samples = rf_sample(model=rf_model, shape=x_start.shape, model_kwargs=model_kwargs, guidance_scale=opt.guidance_scale, device=device, f0=f0)
        print("RF sampling finished!!!")
        
        # DyMeshVAE decoding
        if opt.rescale:
            x0_start_s = samples[:, :, :opt.f0_channels] * x0_std + x0_mean
            xt_start_s = samples[:, :, opt.f0_channels:] * xt_std + xt_mean 
            samples = torch.cat([x0_start_s, xt_start_s], dim=-1)
        outputs = vae_model(vertices, vertices[:, 0], samples=samples, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix, num_traj=opt.num_traj, just_decode=True)

        # assign trajs for each parts
        trajs = [outputs[0][:, idx].cpu() for idx in all_indices]
        # render video
        drive_mesh_with_trajs_frames(mesh_objects, trajs, "{}/{}".format(video_save_dir, filepath.split("/")[-1].split(".")[0]), azi=opt.azi, ele=opt.ele, export_format=opt.export_format)
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./all_glbs")
    parser.add_argument("--vae_dir", type=str, default="./checkpoints")
    parser.add_argument("--rf_model_dir", type=str, default="./checkpoints")
    parser.add_argument("--json_dir", type=str, default="./checkpoints/dvae_factors")
    parser.add_argument("--rf_exp", type=str, default="rf_model")
    parser.add_argument("--rf_epoch", type=str, default='f')
    parser.add_argument("--video_save_dir", type=str, default="./output_videos")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num_traj", type=int, default=-1)
    parser.add_argument("--test_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--azi", type=float, default=0.0)
    parser.add_argument("--ele", type=float, default=0.0)
    parser.add_argument("--export_format", type=str, default="none", choices=["none", "abc", "fbx"])
    
    opt = parser.parse_args()

    opt.rescale = True

    main(opt)

    '''
    python test_drive.py --vae_dir ./checkpoints --rf_model_dir ./checkpoints --json_dir ./checkpoints/dvae_factors --rf_exp rf_model --rf_epoch f --seed 666 --test_name dragon --prompt "The object is flying" --export_format fbx
    '''
   
