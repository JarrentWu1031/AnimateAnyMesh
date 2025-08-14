import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import json
from networks.dymeshvae import DyMeshVAE
from utils.render import render_dynamic_mesh_direct_to_video
from utils.mesh_utils import get_adjacency_matrix

def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    test_data_dir = opt.dataset_dir
    ckpt_dir = os.path.join(opt.ckpt_dir, opt.exp, "dvae_{}.pth".format(opt.epoch))

    all_files = sorted(os.listdir(test_data_dir))
    test_id = [i for i in range(2)]
    files = [all_files[_] for _ in test_id]
    
    video_save_dir = os.path.join(opt.video_save_dir, opt.exp)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    
    config_dir = os.path.join(opt.ckpt_dir, opt.exp, "model_config.json")
    with open(config_dir, 'r') as f:
        model_config = json.load(f)
    model = DyMeshVAE(**model_config).to(device)

    print(f"Loading checkpoint from: {ckpt_dir}")
    checkpoint = torch.load(ckpt_dir, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("  -> Detected new checkpoint format (dictionary).")
        model_weights = checkpoint['model_state_dict']
    else:
        print("  -> Detected old checkpoint format (raw state_dict).")
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    print("Model weights loaded successfully.")

    print(files)
    
    with torch.no_grad():
        test_count = 0 
        avg_vertex_error = 0.0
        for file in files:
            filepath = os.path.join(test_data_dir, file)
            filename = str(test_count)
            assert file.split('.')[-1] == "bin"
            with open(filepath, 'rb') as f:
                mesh_file = pickle.load(f)
                vertices, faces = mesh_file["vertices"], mesh_file["faces"]
                vertices, faces = torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
            
            opt.max_length = max(4096, vertices.shape[1]+512)
            opt.num_traj = max(512, opt.max_length // 8)

            if vertices.shape[1] <= opt.min_length:
                print(vertices.shape)
                print("Too few vertices!!!")
                continue
            if vertices.shape[1] > opt.max_length:
                print(vertices.shape)
                print("Too many vertices!!!")
                continue
    
            center = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
            vertices = vertices - center
            v_max = vertices[0].abs().max()
            vertices = vertices / v_max
            faces_max_length = int(opt.max_length * 2.5)
            assert faces.shape[0] <= faces_max_length
            
            vertices_ori = vertices
            faces_ori = faces

            valid_mask = torch.zeros((1, opt.max_length), dtype=torch.bool, device=device)
            valid_mask[:, :vertices.shape[1]] = True
            valid_length = torch.tensor(vertices_ori.shape[1])[None].to(device)
            vertices = torch.cat([vertices, torch.zeros(vertices.shape[0], opt.max_length-vertices.shape[1], 3)], dim=1)
            faces = torch.cat([faces, -1 * torch.ones(faces_max_length-faces.shape[0], 3).to(torch.int64)], dim=0)
            pc, query, vertices, faces = vertices[None].to(device), vertices[0][None].to(device), vertices.to(device), faces[None].to(device)
            adj_matrix = get_adjacency_matrix(pc[:, 0], faces, valid_length)
            
            output = model(pc, query, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix, num_traj=opt.num_traj)
            recon_pc, pc, idx_temp = output["recon_pc"], output["pc"], output["idx_temp"]
            
            error_rec = recon_pc - pc
            euc_dist = torch.norm(error_rec, p=2, dim=-1)
            masked_euc_dist = euc_dist * valid_mask.unsqueeze(1)
            total_dist_sum = masked_euc_dist.sum(dim=(1, 2)) # 形状 (B,)
            total_valid_observations = pc.shape[1] * valid_mask.sum(dim=1)
            avg_euc_dist = total_dist_sum / (total_valid_observations + 1e-8)

            avg_vertex_error += avg_euc_dist.item()

            test_count += 1
  
            if not os.path.exists(video_save_dir+"/mesh_gt"):
                os.makedirs(video_save_dir+"/mesh_gt")
            if not os.path.exists(video_save_dir+"/mesh_recon"):
                os.makedirs(video_save_dir+"/mesh_recon")
            
            if opt.render:
                print("Start rendering!!!")
                render_dynamic_mesh_direct_to_video(vertices=pc[0].cpu(), face_data=mesh_file["faces"], video_save_dir=video_save_dir+"/mesh_gt", save_name=str(filename), azi=opt.azi, ele=opt.ele)
                render_dynamic_mesh_direct_to_video(vertices=recon_pc[0].cpu(), face_data=mesh_file["faces"], video_save_dir=video_save_dir+"/mesh_recon", save_name=str(filename), azi=opt.azi, ele=opt.ele)
                print("Rendering Ended!!!")
            
        print(opt.exp)
        print("Average vertext reconstruction error: ", avg_vertex_error/len(files))
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="test", choices=["test", "test_color"])
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--ckpt_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--exp", type=str, default="dvae_v1_lr4_avg")
    parser.add_argument("--epoch", type=str, default='f')
    parser.add_argument("--video_save_dir", type=str, default="./output_videos")
    parser.add_argument("--min_length", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--azi", type=float, default=0.0)
    parser.add_argument("--ele", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    
    opt = parser.parse_args()

    if opt.dataset == "test":
        opt.dataset_dir = "./objxl_testset20"

    main(opt)
