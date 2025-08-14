import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPTextModel,
    AutoTokenizer,
)
from .util import timestep_embedding

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class MLP(nn.Module):
    def __init__(self, device, dtype, width, init_scale=0.25):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class QKVMultiheadAttention(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

class QKVMultiheadAttention_flash2(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        
        # 重塑并分离Q,K,V
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        # 调整形状以匹配scaled_dot_product_attention的输入要求 [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 使用scaled_dot_product_attention
        output = F.scaled_dot_product_attention(q, k, v)
        
        # 调整输出形状 [batch_size, seq_length, width]
        output = output.transpose(1, 2).reshape(bs, n_ctx, -1)
        
        return output

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=0.25,
        use_flash2=True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        if use_flash2:
            self.attention = QKVMultiheadAttention_flash2(heads=heads)
        else:
            self.attention = QKVMultiheadAttention(heads=heads)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim, device="cuda", dtype=torch.float32, norm_type="layer_norm", bias=True):
        super().__init__()
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias, device=device)

        self.norm = nn.LayerNorm(embedding_dim, device=device, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class CogXAttentionBlock(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=1.0,
        use_flash2=True,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            use_flash2=use_flash2,
        )
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        
        self.adaln_x = AdaLayerNormZero(width, device=device, dtype=dtype)
        self.adaln_text = AdaLayerNormZero(width, device=device, dtype=dtype)

        self.ln_x = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_text = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x, text_emb, t_emb):
        # token nums
        num_x_token, num_text_token = x.shape[1], text_emb.shape[1]
        
        # adaln for x
        norm_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaln_x(x, emb=t_emb)
        # adaln for text
        norm_text, gate_msa_text, shift_mlp_text, scale_mlp_text, gate_mlp_text = self.adaln_text(text_emb, emb=t_emb)
        
        # concat and pass through attntion layer before split
        norm_all = torch.cat([norm_x, norm_text], dim=1)
        attn_output = self.attn(norm_all)
        attn_output_x, attn_output_text = attn_output[:, :num_x_token], attn_output[:, num_x_token:]
        
        # post process x
        attn_output_x = gate_msa_x.unsqueeze(1) * attn_output_x
        x = x + attn_output_x
        norm_x = self.ln_x(x)
        norm_x = norm_x * (1 + scale_mlp_x[:, None]) + shift_mlp_x[:, None]
        # post process text_embed
        attn_output_text = gate_msa_text.unsqueeze(1) * attn_output_text
        text_emb = text_emb + attn_output_text
        norm_text = self.ln_text(text_emb)
        norm_text = norm_text * (1 + scale_mlp_text[:, None]) + shift_mlp_text[:, None]

        # concat and pass through mlp before split
        ff_output = self.mlp(torch.cat([norm_x, norm_text], dim=1))
        ff_output_x, ff_output_text = ff_output[:, :num_x_token], ff_output[:, num_x_token:]
        
        # final step for x
        ff_output_x = gate_mlp_x[:, None] * ff_output_x
        x = x + ff_output_x
        # final step for text_embed
        ff_output_text = gate_mlp_text[:, None] * ff_output_text
        text_emb = text_emb + ff_output_text

        return x, text_emb

class Transformer_cogx(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        layers,
        heads,
        init_scale=0.25,
        use_flash2=True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                CogXAttentionBlock(
                    device=device,
                    dtype=dtype,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    use_flash2=use_flash2,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor, t_emb: torch.Tensor):
        for block in self.resblocks:
            x, text_emb = block(x, text_emb, t_emb)
        return x

class DyMeshMMDiT(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        input_channels=3*16,
        output_channels=3*16,
        width=512,
        layers=12,
        heads=8,
        init_scale=0.25,
        use_flash2=True,
        cond_drop_prob=0.0,
        **kwargs,
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.backbone = Transformer_cogx(
            device=device,
            dtype=dtype,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            use_flash2=use_flash2,
        )

        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_pre_text = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_token_mlp = nn.Linear(512, self.backbone.width, device=device, dtype=dtype)

        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def _forward_cogx(self, x, text_emb, t_emb):
        h = self.input_proj(x)  
        h = self.ln_pre(h)
        text_emb = self.ln_pre_text(text_emb)
        h = self.backbone(h, text_emb=text_emb, t_emb=t_emb)
        h = self.ln_post(h)
        h = self.output_proj(h)
        return h

    def forward(self, x, t, texts=None):
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        with torch.no_grad():
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.clip_text_model.device)
            text_embed = self.clip_text_model(**text_inputs).last_hidden_state # B, 77, 768
        text_embed = self.clip_token_mlp(text_embed)
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            text_embed = text_embed * mask[:, None, None].to(text_embed)
        else:
            mask = torch.tensor([text != '' for text in texts], dtype=torch.bool)
            text_embed = text_embed * mask[:, None, None].to(text_embed)
        return self._forward_cogx(x, text_embed, t_embed)
