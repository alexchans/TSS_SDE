import math
import torch
import torch.nn as nn
from torchsde import SDEIto

# Bounds on log g. Keep exp() from under/overflowing; wide enough not to bind on
# any physical residual scale (g fits to |resid|/√dt, so g≈10·|resid| at dt=0.01).
LOG_G_MIN, LOG_G_MAX = -8.0, 4.0


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar time t to a vector via sin/cos frequencies + MLP."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        """t: scalar or (B,) → (B, dim)"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class RadialBasisFunction(nn.Module):
    """Encodes distances using evenly-spaced Gaussian radial basis functions."""

    def __init__(self, num_rbf=16, cutoff=30.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        self.width = (centers[1] - centers[0]) * 0.5

    def forward(self, dist):
        """dist: (..., 1) → (..., num_rbf)"""
        return torch.exp(-((dist - self.centers) ** 2) / (2 * self.width ** 2))


class EGNNLayer(nn.Module):
    """E(n)-equivariant message passing layer with attention and FiLM time conditioning."""

    def __init__(self, hidden_dim, num_rbf=16):
        super().__init__()
        self.rbf = RadialBasisFunction(num_rbf=num_rbf)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, h, x, edge_src, edge_dst, time_emb):
        B, N, D = h.shape

        # FiLM: modulate node features with time
        film_params = self.film(time_emb)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        h_src = h[:, edge_src]
        h_dst = h[:, edge_dst]

        rel_pos = x[:, edge_src] - x[:, edge_dst]
        dist = (rel_pos ** 2).sum(-1, keepdim=True).sqrt()

        dist_feat = self.rbf(dist)

        edge_input = torch.cat([h_src, h_dst, dist_feat], dim=-1)
        m_ij = self.edge_mlp(edge_input)

        attn = self.attn_mlp(m_ij)
        m_ij_weighted = m_ij * attn

        coord_weight = self.coord_mlp(m_ij)
        coord_weight = torch.tanh(coord_weight)
        weighted_rel = rel_pos * coord_weight

        idx_3 = edge_dst.view(1, -1, 1).expand(B, -1, 3)
        coord_agg = torch.zeros(B, N, 3, device=x.device, dtype=x.dtype)
        coord_agg.scatter_add_(1, idx_3, weighted_rel)
        x_out = x + coord_agg

        idx_d = edge_dst.view(1, -1, 1).expand(B, -1, D)
        msg_agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        msg_agg.scatter_add_(1, idx_d, m_ij_weighted)

        h_normed = self.layer_norm(h)
        h_out = h + self.node_mlp(torch.cat([h_normed, msg_agg], dim=-1))

        return h_out, x_out


class GeometricSDE(SDEIto):
    """SDE with EGNN drift over a dynamic KNN graph and state-dependent diffusion."""

    def __init__(self, num_residues, hidden_dim=256, num_layers=6, k=8):
        super().__init__(noise_type="diagonal")
        self.num_residues = num_residues
        self.hidden_dim = hidden_dim
        self.k = min(k, num_residues - 1)

        self.node_embedding = nn.Embedding(num_residues, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Per-atom diffusion MLP: node embedding → per-atom, per-axis log-diffusion.
        # Starts uniform at exp(0)=1.0, i.e. a per-step noise of g·√dt = 0.1 Å — the
        # right order for MD frame-to-frame fluctuation, so the NLL starts finite.
        # Atoms differentiate once gradients flow through their node embeddings.
        self.diffusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.diffusion_mlp[-1].weight)
        nn.init.zeros_(self.diffusion_mlp[-1].bias)

    @torch.no_grad()
    def _build_knn_edges(self, x):
        """Build bidirectional KNN graph from positions x: (B, N, 3)."""
        x_mean = x.mean(dim=0)
        dists = torch.cdist(x_mean.unsqueeze(0), x_mean.unsqueeze(0)).squeeze(0)
        dists.fill_diagonal_(float('inf'))

        _, knn_idx = dists.topk(self.k, largest=False, dim=-1)

        N = x.shape[1]
        src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.k).reshape(-1)
        dst = knn_idx.reshape(-1)

        # Make bidirectional (add reverse edges) and deduplicate
        edge_src = torch.cat([src, dst])
        edge_dst = torch.cat([dst, src])
        edge_pairs = torch.stack([edge_src, edge_dst], dim=0)
        edge_pairs = torch.unique(edge_pairs, dim=1)
        return edge_pairs[0], edge_pairs[1]

    def set_edges(self, y):
        """Cache KNN edges for the current batch. Call before sdeint."""
        B = y.shape[0]
        x = y.view(B, self.num_residues, 3)
        self._cached_edges = self._build_knn_edges(x)

    def f(self, t, y):
        """Equivariant drift function with time conditioning."""
        B = y.shape[0]
        x = y.view(B, self.num_residues, 3)
        x0 = x

        h = self.node_embedding.weight.unsqueeze(0).expand(B, -1, -1)

        if hasattr(self, '_cached_edges') and self._cached_edges is not None:
            edge_src, edge_dst = self._cached_edges
        else:
            edge_src, edge_dst = self._build_knn_edges(x)

        t_emb = self.time_embed(t.expand(B))

        for layer in self.layers:
            h, x = layer(h, x, edge_src, edge_dst, t_emb)

        drift = x - x0
        return drift.reshape(B, -1)

    def g(self, t, y):
        """Per-atom diagonal diffusion from node embeddings.

        The MLP emits log g, so g is strictly positive and unbounded above. A
        sigmoid ceiling would clip g below the residual scale the NLL objective
        fits it to — the old sigmoid * 1/√N form capped g at ≈0.18 for N=31,
        which no per-step MD fluctuation fits under.
        """
        B = y.shape[0]
        # Each atom gets its own diffusion coefficient from its embedding
        h = self.node_embedding.weight  # (N, hidden_dim)
        log_diff = self.diffusion_mlp(h).clamp(LOG_G_MIN, LOG_G_MAX)  # (N, 3)
        diff = torch.exp(log_diff)
        diff = diff.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        return diff.reshape(B, -1)  # (B, N*3)
