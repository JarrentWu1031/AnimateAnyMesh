import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch import einsum
import numpy as np
from functools import wraps
from timm.models.layers import DropPath
import pytorch3d.ops as ops
from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm_qkv(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, fn, norm_type="qkv"):
        super().__init__()
        self.fn = fn
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(value_dim)
        self.norm_type = norm_type

    def forward(self, x, key, value, **kwargs):
        if self.norm_type == "q":
            x = self.norm_q(x)
        elif self.norm_type == "qk":
            x = self.norm_q(x)
            key = self.norm_k(key)
        elif self.norm_type == "qkv":
            x = self.norm_q(x)
            key = self.norm_k(key)
            value = self.norm_v(value)
        else:
            raise ValueError("Unsupported norm type!!!")
        return self.fn(x, key, value, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class Attention_qkv(nn.Module):
    def __init__(self, query_dim, key_dim = None, value_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        
        key_dim = default(key_dim, query_dim)
        value_dim = default(value_dim, key_dim) 
        
        inner_dim = dim_head * heads
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, query_dim)

        self._zero_init_out_layer()

    def _zero_init_out_layer(self):
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, x, key=None, value=None, mask=None):
        h = self.heads
        key = default(key, x)
        value = default(value, key)

        # proj
        q = self.to_q(x)
        k = self.to_k(key)
        v = self.to_v(value)

        # rearrange
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # mask
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]

        # attention
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
            ) 

        # rearrange
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

class TrajEncoding(nn.Module):
    def __init__(self, out_dim, num_freqs=10, include_input=True, log_sampling=True, input_dims=48):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0, 2. ** (num_freqs - 1), num_freqs)
        self.register_buffer('freq_bands', freq_bands.view(-1, 1))
        latent_dim = self.input_dims * (2 * self.num_freqs + int(self.include_input))
        self.mlp = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        assert x.dim() == 3 and x.shape[-1] == self.input_dims, \
            f"Expected input shape [B, N, {self.input_dims}], but got {x.shape}"
        x_expanded = x.unsqueeze(2)
        x_freq = x_expanded * (self.freq_bands.view(1, 1, -1, 1) * torch.pi)
        sin_feat = torch.sin(x_freq)  # [B, N, num_freqs, input_dims]
        cos_feat = torch.cos(x_freq)  # [B, N, num_freqs, input_dims]
        sin_feat = sin_feat.reshape(x.shape[0], x.shape[1], -1)  # [B, N, num_freqs*input_dims]
        cos_feat = cos_feat.reshape(x.shape[0], x.shape[1], -1)  # [B, N, num_freqs*input_dims]
        return self.mlp(torch.cat([x, sin_feat, cos_feat], dim=-1))  # [B, N, input_dims + 2*num_freqs*input_dims]

class JointPreNorm(nn.Module):
    def __init__(self, dim1, dim2, fn, context_dim=None, norm_type="qkv"):
        super().__init__()
        self.fn = fn
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)
        if context_dim is not None:
            self.norm_context = nn.LayerNorm(context_dim)
        self.context_dim = context_dim
        self.norm_type = norm_type

    def forward(self, x1, x2, context=None, **kwargs):
        if self.norm_type == "q":
            q1 = self.norm1(x1)
            k1 = v1 = default(context, x1)
            v2 = x2
        elif self.norm_type == "qk":
            q1 = self.norm1(x1)
            if context is not None:
                k1 = v1 = context
                k1 = self.norm_context(k1)
            else:
                k1 = q1
                v1 = x1
            v2 = x2
        elif self.norm_type == "qkv":
            q1 = self.norm1(x1)
            if context is not None:
                k1 = v1 = self.norm_context(context)
            else:
                k1 = v1 = q1
            v2 = self.norm2(x2)
        else:
            raise ValueError("Unsupported norm type!!!")
        return self.fn(q1, k1, v1, v2, **kwargs)

class JointPreNorm_ffn(nn.Module):
    def __init__(self, dim1, dim2, fn):
        super().__init__()
        self.fn = fn
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)

    def forward(self, x1, x2, **kwargs):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        return self.fn(x1, x2, **kwargs)

class JointFeedForward(nn.Module):
    def __init__(self, dim1, dim2, mult = 4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim1, dim1 * mult * 2),
            GEGLU(),
            nn.Linear(dim1 * mult, dim1)
        )
        self.net2 = nn.Sequential(
            nn.Linear(dim2, dim2 * mult * 2),
            GEGLU(),
            nn.Linear(dim2 * mult, dim2)
        )

    def forward(self, x1, x2):
        return self.net1(x1), self.net2(x2)


class JointAttention(nn.Module):
    def __init__(self, dim1, dim2, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.to_q1 = nn.Linear(dim1, inner_dim, bias=False)
        self.to_k1 = nn.Linear(dim1, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim1, inner_dim, bias=False)
        self.to_out1 = nn.Linear(inner_dim, dim1)
        self.to_v2 = nn.Linear(dim2, inner_dim, bias=False)        
        self.to_out2 = nn.Linear(inner_dim, dim2)

        self._zero_init()
    
    def _zero_init(self):
        nn.init.zeros_(self.to_out1.weight)
        nn.init.zeros_(self.to_out1.bias)
        nn.init.zeros_(self.to_out2.weight)
        nn.init.zeros_(self.to_out2.bias)

    def forward(self, q1, k1, v1, v2, mask=None):
        h = self.heads
        
        # ln
        q1 = self.to_q1(q1)
        k1 = self.to_k1(k1)
        v1 = self.to_v1(v1)
        v2 = self.to_v2(v2)

        # rearrange
        q1, k1, v1, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q1, k1, v1, v2))
        
        q1 = q1  # [B, H, N, D]
        k1 = k1  # [B, H, N, D]
        v1_v2 = torch.cat([v1, v2], dim=-1)  # [B, 2H, N, D]

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]
        
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(
                q1, k1, v1_v2,
                attn_mask=mask,
            )  # [B, H, N, 2D]
        out1, out2 = out.chunk(2, dim=-1)  

        out1, out2 = map(lambda t: rearrange(t, 'b h n d -> b n (h d)', h = h), (out1, out2))
        out1 = self.to_out1(out1)
        out2 = self.to_out2(out2)

        return out1, out2

class NeighborAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, value_dim)
        
        self._zero_init()
    
    def _zero_init(self):
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, k, v, mask = None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None] # b, h, n, n
            else:
                assert mask.shape[1] == h
            if mask.dtype != torch.bool:
                mask = torch.log(mask + 1e-9)
        
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_output = scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        neighbor_features = rearrange(attn_output, 'b h n d -> b n (h d)', h=h)
        neighbor_features = self.to_out(neighbor_features)
        
        return neighbor_features

def replace_negative_indices(idx, valid_len):
    B, M = idx.shape
    device = idx.device
    neg_mask = idx <= 0  # [B, M]
    rand_values = torch.rand(B, M, device=device)  # [B, M]
    valid_len_expanded = valid_len.unsqueeze(1).expand(-1, M)  # [B, M]
    random_indices = (rand_values * valid_len_expanded.float()).long()
    new_idx = torch.where(neg_mask, random_indices, idx)
    return new_idx

class DyMeshVAE(nn.Module):
    def __init__(
        self,
        *,
        enc_depth=8,
        dec_depth=8,
        dim=256,
        output_dim=3*16,
        latent_dim=32,
        heads=8,
        dim_head=64,
        T=16,
        num_traj=512, 
        n_layers=1,
        norm_type="qkv",
    ):
        super().__init__()

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.num_traj = num_traj
        self.T = T
        self.latent_dim = latent_dim
        
        # layer definition
        get_dymesh_neighbor_attn = lambda: PreNorm_qkv(dim, dim, dim, NeighborAttention(query_dim=dim, key_dim=dim, value_dim=dim, heads=1, dim_head=dim*2), norm_type=norm_type)
        get_dymesh_attn_enc = lambda: JointPreNorm(dim, dim, JointAttention(dim1=dim, dim2=dim, heads=1, dim_head=dim), context_dim=dim, norm_type=norm_type)
        get_dymesh_attn_dec = lambda: JointPreNorm(dim, dim, JointAttention(dim1=dim, dim2=dim, heads=heads, dim_head=dim_head), context_dim=dim, norm_type=norm_type)
        get_dymesh_ffn = lambda: JointPreNorm_ffn(dim, dim, JointFeedForward(dim1=dim, dim2=dim))

        ### Encoder
        # vertex embed & relative trajectory embed
        self.point_embed = PointEmbed(dim=dim)
        self.traj_embed = TrajEncoding(out_dim=dim, input_dims=3*T)
        # adjacent info layers
        self.neighbor_layers = nn.ModuleList([])
        self.n_layers = n_layers
        for i in range(n_layers):
            self.neighbor_layers.append(get_dymesh_neighbor_attn())
        # enc cross attn & ffn
        self.enc_depth = enc_depth
        self.enc_blocks = nn.ModuleList([])
        for i in range(enc_depth):
            self.enc_blocks.append(nn.ModuleList([
                    get_dymesh_attn_enc(),
                    get_dymesh_ffn(),
                ]))
        
        ### VAE sample
        # x0 projection
        self.fc_x0 = nn.Linear(dim, latent_dim)
        # xt vae sampler
        self.mean_fc_xt = nn.Linear(dim, latent_dim)
        self.logvar_fc_xt = nn.Linear(dim, latent_dim)
        
        ### Decoder
        # x0 & xt projector
        self.proj_x0 = nn.Linear(latent_dim, dim)
        self.proj_xt = nn.Linear(latent_dim, dim)
        # dec cross attn & ffn
        self.dec_depth = dec_depth
        self.dec_blocks = nn.ModuleList([])
        for i in range(dec_depth):
            self.dec_blocks.append(nn.ModuleList([
                get_dymesh_attn_dec(),
                get_dymesh_ffn(),
            ]))
        self.fc_query = nn.Linear(dim, dim*2)
        self.decoder_final_ca = PreNorm_qkv(dim*2, dim, dim, Attention_qkv(dim*2, dim, dim, heads = 1, dim_head = dim*2))
        self.decoder_final_ffn = nn.Linear(dim*2, output_dim) 

    def encode(self, pc, faces=None, valid_mask=None, adj_matrix=None):
        B, T, N, D = pc.shape
        device = pc.device
        
        pct = pc
        pc0 = pc[:, 0] # B, N, C
        pct_rel = pct - pct[:, :1]
        pct_rel_ori = pct_rel
        pct_rel = pct_rel.permute(0, 2, 1, 3).flatten(2, 3) # B, N, TD
        pc0_embed = self.point_embed(pc0)  
        pct_embed = self.traj_embed(pct_rel)
        pc0_ori = pc0

        self.pc0_embed_pe = pc0_embed

        if adj_matrix is not None:
            for i in range(self.n_layers):
                neighbor_layer = self.neighbor_layers[i]
                pc0_embed_res = neighbor_layer(pc0_embed, key=pc0_embed, value=pc0_embed, mask=adj_matrix)
                pc0_embed = pc0_embed + pc0_embed_res
        
        pc0_embed_ori = pc0_embed
        pct_embed_ori = pct_embed
        
        # sample representative trajs
        num_traj = self.num_traj
        with torch.no_grad():
            if valid_mask is not None:
                valid_length = valid_mask.sum(dim=-1)
                _, idx = ops.sample_farthest_points(points=pc0_embed, lengths=valid_length, K=num_traj)
                idx = replace_negative_indices(idx, valid_length)
            else:
                _, idx = ops.sample_farthest_points(points=pc0_embed, K=self.num_traj)
                valid_length = torch.tensor([N], dtype=torch.int64, device=pc.device)
                idx = replace_negative_indices(idx, valid_length)
        
        for i in range(self.enc_depth):
            if i == 0:
                pc0_embed_gathered = torch.gather(pc0_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pc0_embed.shape[-1]))
                pct_embed_gathered = torch.gather(pct_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pct_embed.shape[-1]))
                valid_selection_mask = (idx < valid_length.unsqueeze(1))  # 形状: (B, K), True 表示有效选择
                mask = valid_selection_mask.unsqueeze(-1).float()
                pc0_embed = pc0_embed_gathered * mask
                pct_embed = pct_embed_gathered * mask
            dymesh_attn, dymesh_ffn = self.enc_blocks[i]
            # attn
            attn_out0, attn_outT = dymesh_attn(pc0_embed, pct_embed_ori, context=pc0_embed_ori)
            pc0_embed = pc0_embed + attn_out0
            pct_embed = pct_embed + attn_outT
            # ffn
            ffn_out0, ffn_outT = dymesh_ffn(pc0_embed, pct_embed)
            pc0_embed = pc0_embed + ffn_out0
            pct_embed = pct_embed + ffn_outT
        
        # x0
        x0 = pc0_embed
        x0 = self.fc_x0(x0)
        # xt
        xt = pct_embed
        # xt vae sample
        mean = self.mean_fc_xt(xt)
        logvar = self.logvar_fc_xt(xt)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        xt = posterior.sample()
        # kl divergence
        kl_xt = posterior.kl()
        kl = kl_xt
        x = torch.cat([x0, xt], dim=-1)

        return kl, x, idx, pc0_embed_ori

    def decode(self, x, queries, pc0_embed, add_adj=False, adj_matrix=None):
        device = x.device
        
        x0, xt = x[:, :, :self.latent_dim], x[:, :, self.latent_dim:]
        x0 = self.proj_x0(x0)
        xt = self.proj_xt(xt)

        for dymesh_attn, dymesh_ffn in self.dec_blocks:
            # attn
            attn_out0, attn_outT = dymesh_attn(x0, xt)
            x0 = x0 + attn_out0
            xt = xt + attn_outT
            # ffn
            ff_out0, ff_outT = dymesh_ffn(x0, xt)
            x0 = x0 + ff_out0
            xt = xt + ff_outT

        B, N, C = queries.shape
        queries_embeddings = self.fc_query(pc0_embed)
        latents = self.decoder_final_ca(queries_embeddings, key = x0, value = xt)
        outputs = self.decoder_final_ffn(latents)
        outputs = outputs.view(B, queries.shape[1], -1, 3).permute(0, 2, 1, 3) # B, T, N, C
        outputs = queries[:, None].repeat(1, self.T, 1, 1) + outputs
        
        return outputs

    def forward(self, pc, queries, faces=None, valid_mask=None, adj_matrix=None, num_traj=None, just_encode=False, just_decode=False, samples=None):
        if num_traj is not None:
            self.num_traj = num_traj
            
        kl, x, idx, pc0_embed_ori = self.encode(pc, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix)

        if just_encode:
            return x
        
        if just_decode:
            recon_pc = self.decode(samples, queries, pc0_embed_ori, adj_matrix=adj_matrix)
            return recon_pc
        else:
            recon_pc = self.decode(x, queries, pc0_embed_ori, adj_matrix=adj_matrix)

        return {'recon_pc': recon_pc, "pc": pc, 'kl_temp': kl, 'idx_temp': idx, 'latent': x, 'pc0_embed_ori': pc0_embed_ori, 'pc0_embed_pe': self.pc0_embed_pe}


if __name__ == "__main__": 
    import pickle 
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from utils.mesh_utils import get_adjacency_matrix

    device = torch.device("cuda")
    model = DyMeshVAE(
                enc_depth=4,
                norm_type="qkv"
            ).to(device)