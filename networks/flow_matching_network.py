import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import NETWORK_REGISTRY


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, max_freq=100.0):
        super().__init__()
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim
        half = emb_dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), half))  # [half]
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # [B, half]
        x = t * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, t_dim: int, c: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(t_dim, 2 * c)

        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # h: [B,S,C], t_emb: [B,t_dim]
        B, S, C = h.shape
        gb = self.to_gamma_beta(t_emb)           # [B,2C]
        gamma, beta = gb[:, :C], gb[:, C:]       # [B,C], [B,C]
        gamma = gamma[:, None, :].expand(B, S, C)
        beta  = beta[:, None, :].expand(B, S, C)
        return h * (1.0 + gamma) + beta

@NETWORK_REGISTRY.register()
class FlowField(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 512, depth: int = 4, time_emb_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.in_proj = nn.Linear(in_channels, hidden_dim)

        films = []
        blocks = []
        for _ in range(depth - 1):
            blocks.append(nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)))
            films.append(FiLM(time_emb_dim, hidden_dim))

        self.blocks = nn.ModuleList(blocks)
        self.films = nn.ModuleList(films)

        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, in_channels))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: [B,S,C]
        t:   [B,1]
        """
        t_emb = self.time_emb(t)
        h = self.in_proj(z_t)                   # [B,S,Dt]

        for blk, film in zip(self.blocks, self.films):
            h = film(h, t_emb)        # time modulation
            h = blk(h)                # nonlinearity + linear
        v = self.out(h)               # [B,S,C]
        return v


def gather_neigh(x, idx):
    B, N, C = x.shape
    _, _, K = idx.shape
    offset = (torch.arange(B, device=x.device) * N).view(B, 1, 1)  # [B,1,1]
    idx_flat = (idx + offset).reshape(-1)                          # [B*N*K]
    x_flat = x.reshape(B * N, C)                                   # [B*N, C]
    out = x_flat.index_select(0, idx_flat).reshape(B, N, K, C)     # [B,N,K,C]
    return out


@NETWORK_REGISTRY.register()
class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        geo_dim: int = 256,
        sem_dim: int = 768,
        attn_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        gate_alpha: float = 1.0,
        use_pos_bias: bool = False,
        pos_bias_hidden: int = 64,
        add_ffn: bool = True,
        ffn_hidden_mult: int = 4,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.geo_dim = geo_dim
        self.sem_dim = sem_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.dropout = dropout
        self.gate_alpha = gate_alpha
        self.use_pos_bias = use_pos_bias
        self.add_ffn = add_ffn

        # --- semantic projection 768 -> 256 ---
        self.sem_to_attn = nn.Sequential(
            nn.Linear(sem_dim, attn_dim),
            nn.LayerNorm(attn_dim),
        )

        # --- vertex-wise channel gating: semantic -> geo gate ---
        self.gate = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, geo_dim),
        )

        # --- QKV projections ---
        self.q_proj = nn.Linear(geo_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, geo_dim, bias=False)

        # --- norms ---
        self.norm_geo = nn.LayerNorm(geo_dim)
        self.norm_sem = nn.LayerNorm(attn_dim)
        self.norm_attn = nn.LayerNorm(attn_dim)
        self.norm_ffn = nn.LayerNorm(attn_dim) if add_ffn else None

        # --- optional FFN ---
        if add_ffn:
            hidden = ffn_hidden_mult * attn_dim
            self.ffn = nn.Sequential(
                nn.Linear(attn_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, attn_dim),
                nn.Dropout(dropout),
            )

    def forward(self, f_geo: torch.Tensor, f_sem: torch.Tensor, neigh: torch.Tensor, verts: torch.Tensor | None = None):
        neigh = neigh.unsqueeze(0) # add batch dim to neighbor indices for gather
        B, N, Cg = f_geo.shape
        _, N2, Cs = f_sem.shape
        assert N2 == N and Cg == self.geo_dim and Cs == self.sem_dim
        _, _, K = neigh.shape

        # --- project semantic ---
        sem = self.norm_sem(self.sem_to_attn(f_sem))  # [B,N,attn_dim]

        # --- vertex-wise gating ---
        geo = self.norm_geo(f_geo)                    # [B,N,geo_dim]
        gate = torch.sigmoid(self.gate(sem))            # [B,N,geo_dim]
        geo_gated = geo * (1.0 + self.gate_alpha * gate)

        # --- Q from geo, K/V from semantic ---
        Q = self.q_proj(geo_gated)                      # [B,N,attn_dim]
        K_all = self.k_proj(sem)                        # [B,N,attn_dim]
        V_all = self.v_proj(sem)                        # [B,N,attn_dim]

        # reshape to heads
        Qh = Q.view(B, N, self.num_heads, self.head_dim)             # [B,N,H,D]
        Qh = Qh.permute(0, 2, 1, 3).contiguous()                     # [B,H,N,D]

        # gather neighbor K/V: [B,N,K,attn_dim] -> [B,H,N,K,D]
        K_nbr = gather_neigh(K_all, neigh)               # [B,N,K,C]
        V_nbr = gather_neigh(V_all, neigh)               # [B,N,K,C]
        K_nbr = K_nbr.view(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        V_nbr = V_nbr.view(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        # K_nbr, V_nbr: [B,H,N,K,D]

        # --- attention logits: dot(Q_i, K_{i,j}) over neighbor j ---
        # logits: [B,H,N,K]
        logits = (Qh.unsqueeze(3) * K_nbr).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = torch.softmax(logits, dim=-1)                         # [B,H,N,K]
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # --- weighted sum: [B,H,N,D] ---
        out_h = (attn.unsqueeze(-1) * V_nbr).sum(dim=3)              # [B,H,N,D]

        # merge heads
        out = out_h.permute(0, 2, 1, 3).contiguous().view(B, N, self.attn_dim)  # [B,N,C]
        out = self.norm_attn(out)

        # optional FFN in attn space
        if self.add_ffn:
            out = out + self.ffn(self.norm_ffn(out))

        # fused = f_geo + 0.01 * self.out_proj(out)                         # residual back to geo_dim
        fused = f_geo + self.out_proj(out)  # original without scaling
        return fused
