import torch


def get_neigh(faces: torch.Tensor, num_verts: int, K: int = 32, add_self: bool = True):
    device = faces.device
    faces = faces.squeeze(0)  # [1,F,3] -> [F,3]
    faces = faces.long()

    # 1) build directed edges (u->v and v->u), total 6F
    edges = torch.cat((faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]), dim=0)              # [3F,2]
    edges_rev = edges[:, [1, 0]]
    edges = torch.cat([edges, edges_rev], dim=0)           # [6F,2]

    # 2) remove self edges
    edges = edges[edges[:, 0] != edges[:, 1]]

    # 3) add self-loops (i->i)
    if add_self:
        self_edges = torch.arange(num_verts, device=device).view(-1, 1).repeat(1, 2)  # [N,2] (i,i)
        edges = torch.cat([edges, self_edges], dim=0)

    # 4) sort by (src, self_first, dst) so self-loop comes first per vertex
    src = edges[:, 0]
    dst = edges[:, 1]
    is_self = (src == dst).long()  # 1 if self
    secondary = (1 - is_self)      # 0 for self, 1 for non-self

    # key = src * (2 * (num_verts + 1)) + secondary * (num_verts + 1) + dst
    key = src.to(torch.int64) * (2 * (num_verts + 1)) + secondary.to(torch.int64) * (num_verts + 1) + dst.to(torch.int64)
    order = torch.argsort(key)
    src = src[order]
    dst = dst[order]

    # 5) remove duplicate (src, dst) pairs (now consecutive after sort)
    if src.numel() > 1:
        same_as_prev = (src[1:] == src[:-1]) & (dst[1:] == dst[:-1])
        keep = torch.cat([torch.ones(1, device=device, dtype=torch.bool), ~same_as_prev], dim=0)
        src = src[keep]
        dst = dst[keep]

    # 6) build fixed-size neighbor list without python loop
    nbr_idx = torch.arange(num_verts, device=device).view(-1, 1).repeat(1, K)

    counts = torch.bincount(src, minlength=num_verts)
    offsets = torch.zeros(num_verts + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # rank within each src segment
    rank = torch.arange(src.numel(), device=device, dtype=torch.long) - offsets[src]
    mask = rank < K

    nbr_idx[src[mask], rank[mask]] = dst[mask]

    return nbr_idx
