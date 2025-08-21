import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops as ops
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm_qkv(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, fn):
        super().__init__()
        self.fn = fn
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(value_dim)

    def forward(self, x, key, value, **kwargs):
        x = self.norm_q(x)
        key = self.norm_k(key)
        value = self.norm_v(value)
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

class SyncAttentionPreNorm(nn.Module):
    def __init__(self, dim, attn_module):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_v2 = nn.LayerNorm(dim)
        self.attn = attn_module

    def forward(self, q_stream, k_stream, v1_stream, v2_stream):
        q = self.norm_q(q_stream)
        k = self.norm_k(k_stream)
        v1 = self.norm_v1(v1_stream)
        v2 = self.norm_v2(v2_stream)
        return self.attn(q, k, v1, v2)

class SyncFFNPreNorm(nn.Module):
    def __init__(self, dim, ffn_module):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ffn_module

    def forward(self, x1, x2):
        return self.ffn(self.norm1(x1), self.norm2(x2))
    
class SyncAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out1 = nn.Linear(inner_dim, dim)
        self.to_out2 = nn.Linear(inner_dim, dim)

    def forward(self, q_stream, k_stream, v1_stream, v2_stream):
        h = self.heads
        q = self.to_q(q_stream)
        k = self.to_k(k_stream)
        v1 = self.to_v1(v1_stream)
        v2 = self.to_v2(v2_stream)
        q, k, v1, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v1, v2))
        v_cat = torch.cat([v1, v2], dim=-1)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out_cat = F.scaled_dot_product_attention(q, k, v_cat)
        out1, out2 = out_cat.chunk(2, dim=-1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        return self.to_out1(out1), self.to_out2(out2)

class SyncFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x1, x2):
        return self.ffn1(x1), self.ffn2(x2)

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
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
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
    ):
        super().__init__()

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.num_traj = num_traj
        self.T = T
        self.latent_dim = latent_dim
        
        # Layer definition
        get_neighbor_attn = lambda: PreNorm_qkv(dim, dim, dim, NeighborAttention(query_dim=dim, key_dim=dim, value_dim=dim, heads=1, dim_head=2*dim))
        get_enc_attn = lambda: SyncAttentionPreNorm(dim, SyncAttention(dim=dim, heads=1, dim_head=dim))
        get_enc_ffn = lambda: SyncFFNPreNorm(dim, SyncFeedForward(dim=dim))
        get_dec_attn = lambda: SyncAttentionPreNorm(dim, SyncAttention(dim=dim, heads=heads, dim_head=dim_head))
        get_dec_ffn = lambda: SyncFFNPreNorm(dim, SyncFeedForward(dim=dim))

        ### Encoder
        self.point_embed = PointEmbed(dim=dim)
        self.traj_embed = TrajEncoding(out_dim=dim, input_dims=3*T)
        # Adjacency info layers
        self.neighbor_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.neighbor_layers.append(get_neighbor_attn())
        # Encoder attn & ffn
        self.enc_blocks = nn.ModuleList([])
        for _ in range(enc_depth):
            self.enc_blocks.append(nn.ModuleList([
                get_enc_attn(),
                get_enc_ffn(),
            ]))
        
        ### VAE sample
        self.mean_fc_x0 = nn.Linear(dim, latent_dim)
        self.mean_fc_xt = nn.Linear(dim, latent_dim)
        self.logvar_fc_xt = nn.Linear(dim, latent_dim)
        
        ### Decoder
        # projector
        self.proj_x0 = nn.Linear(latent_dim, dim)
        self.proj_xt = nn.Linear(latent_dim, dim)
        # Decoder attn & ffn
        self.dec_blocks = nn.ModuleList([])
        for _ in range(dec_depth):
            self.dec_blocks.append(nn.ModuleList([
                get_dec_attn(), 
                get_dec_ffn(),
            ]))
        # Output layers
        self.fc_query = nn.Linear(dim, dim*2)
        self.decoder_final_ca = PreNorm_qkv(dim*2, dim, dim, Attention_qkv(dim*2, dim, dim, heads=1, dim_head=dim*2))
        self.to_outputs = nn.Linear(dim*2, output_dim)

    def encode(self, pc, faces=None, valid_mask=None, adj_matrix=None):
        B, T, N, D = pc.shape
        device = pc.device
        
        # Reshape input
        pct = pc
        pc0 = pc[:, 0]
        pct_rel = (pc - pc[:, :1]).permute(0, 2, 1, 3).flatten(2, 3)
        pc0_embed = self.point_embed(pc0)
        pct_embed = self.traj_embed(pct_rel)
        
        # Adj info aggregation
        if adj_matrix is not None:
            for neighbor_layer in self.neighbor_layers:
                pc0_embed_res = neighbor_layer(pc0_embed, key=pc0_embed, value=pc0_embed, mask=adj_matrix)
                pc0_embed = pc0_embed + pc0_embed_res
        pc0_embed_ori = pc0_embed
        pct_embed_ori = pct_embed

        # Sample & Gather
        with torch.no_grad():
            valid_length = valid_mask.sum(dim=-1)
            _, idx = ops.sample_farthest_points(points=pc0_embed, lengths=valid_length, K=self.num_traj)
            idx = replace_negative_indices(idx, valid_length)
        pc0_embed = torch.gather(pc0_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pc0_embed.shape[-1]))
        pct_embed = torch.gather(pct_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pct_embed.shape[-1]))

        # Enc CA & FFN
        for enc_attn, enc_ffn in self.enc_blocks:
            # CA
            attn_res_0, attn_res_t = enc_attn(
                q_stream=pc0_embed, 
                k_stream=pc0_embed_ori, 
                v1_stream=pc0_embed_ori, 
                v2_stream=pct_embed_ori
            )
            pc0_embed = pc0_embed + attn_res_0
            pct_embed = pct_embed + attn_res_t
            # FFN
            ffn_res_0, ffn_res_t = enc_ffn(pc0_embed, pct_embed)
            pc0_embed = pc0_embed + ffn_res_0
            pct_embed = pct_embed + ffn_res_t
        
        # VAE
        x0 = self.mean_fc_x0(pc0_embed)
        mean = self.mean_fc_xt(pct_embed)
        logvar = self.logvar_fc_xt(pct_embed)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        xt = posterior.sample()
        kl = posterior.kl()
        x = torch.cat([x0, xt], dim=-1)

        return kl, x, idx, pc0_embed_ori

    def decode(self, x, queries, pc0_embed_ori):
        # Projection
        x0_latent, xt_latent = x.chunk(2, dim=-1)
        x0 = self.proj_x0(x0_latent)
        xt = self.proj_xt(xt_latent)
        
        # Dec SA & FFN
        for dec_attn, dec_ffn in self.dec_blocks:
            # SA
            attn_res_0, attn_res_t = dec_attn(
                q_stream=x0, 
                k_stream=x0, 
                v1_stream=x0, 
                v2_stream=xt
            )
            x0 = x0 + attn_res_0
            xt = xt + attn_res_t
            # FFN
            ffn_res_0, ffn_res_t = dec_ffn(x0, xt)
            x0 = x0 + ffn_res_0
            xt = xt + ffn_res_t

        # Final CA & Projection
        query_embed = self.fc_query(pc0_embed_ori)
        latents = self.decoder_final_ca(query_embed, key=x0, value=xt)
        outputs = self.to_outputs(latents)
        outputs = outputs.view(x.shape[0], queries.shape[1], -1, 3).permute(0, 2, 1, 3)
        outputs = queries[:, None] + outputs
        
        return outputs

    def forward(self, pc, queries, faces=None, valid_mask=None, adj_matrix=None, num_traj=None, just_encode=False, just_decode=False, samples=None):
        if num_traj is not None:
            self.num_traj = num_traj
            
        kl, x, idx, pc0_embed_ori = self.encode(pc, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix)

        if just_encode: return x
        
        recon_pc = self.decode(samples if just_decode else x, queries, pc0_embed_ori)

        if just_decode: return recon_pc

        return {'recon_pc': recon_pc, "pc": pc, 'kl_temp': kl, 'idx_temp': idx, 'latent': x}