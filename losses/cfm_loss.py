import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F

@LOSS_REGISTRY.register()
class FlowMatchingLoss(nn.Module):
    def __init__(self, sample_ratio=0.5, alpha=2, detach_pi=False, normalize_feat=True, loss_weight=100, eps=1e-8, charbonnier_eps=1e-3):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.alpha = alpha
        self.detach_pi = detach_pi
        self.normalize_feat = normalize_feat
        self.loss_weight = loss_weight
        self.eps = eps
        self.charbonnier_eps = charbonnier_eps


    def forward(self, flow_field, hx, hy, Pxy, t_flow):
        B, Nx, C = hx.shape
        _, Ny, Cy = hy.shape
        assert Cy == C

        S = min(int(Nx * self.sample_ratio), int(Ny * self.sample_ratio))

        if self.normalize_feat:
            hx_n = F.normalize(hx, dim=-1, p=2)
            hy_n = F.normalize(hy, dim=-1, p=2)
        else:
            hx_n, hy_n = hx, hy

        Pi_full = Pxy.detach() if self.detach_pi else Pxy
        z0_full = hx_n  # [B,Nx,C]
        z1_full = torch.bmm(Pi_full, hy_n)  # [B,Nx,C]

        with torch.no_grad():
            sim = F.cosine_similarity(z0_full, z1_full, dim=-1, eps=self.eps)  # [B,Nx]
            weights = torch.exp(self.alpha * sim)  # [B,Nx]
            weights = torch.nan_to_num(weights, nan=0.0)

        idx = torch.multinomial(weights, num_samples=S, replacement=False)

        z0 = z0_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))  # [B,S,C]
        z1 = z1_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))  # [B,S,C]

        # CFM path
        t = t_flow[:, None, :].expand(B, S, 1)  # [B,S,1]
        z_t = (1.0 - t) * z0 + t * z1                # [B,S,C]
        v_target = (z1 - z0)                         # [B,S,C]

        # predict
        v_pred = flow_field(z_t, t_flow)              # [B,S,C]
        diff = v_pred - v_target
        l_cfm = torch.sqrt(diff * diff + self.charbonnier_eps ** 2 + 1e-9).mean() * self.loss_weight

        return l_cfm