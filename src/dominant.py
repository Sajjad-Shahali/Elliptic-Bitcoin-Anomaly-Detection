import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DOMINANT(nn.Module):
    """
    DOMINANT: Deep Anomaly Detection on Attributed Networks (Ding et al. SDM 2019).
    2-layer GCN encoder → attribute decoder + inner-product structure decoder.
    2 layers only — avoids over-smoothing on sparse graph (avg degree ~2.3).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.attr_decoder = nn.Linear(latent_dim, input_dim)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)  # Z: (n, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        z = self.encode(x, edge_index)
        x_hat = self.attr_decoder(z)
        return z, x_hat


def dominant_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    alpha: weight on attribute loss; (1-alpha) on structure loss.
    Structure decoder: inner-product similarity with negative sampling.
    """
    attr_loss = F.mse_loss(x_hat, x)

    neg_edge = negative_sampling(edge_index, num_nodes=num_nodes,
                                 num_neg_samples=edge_index.size(1), method="sparse")
    pos_sim = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    neg_sim = (z[neg_edge[0]]   * z[neg_edge[1]]).sum(dim=1)

    struct_loss = (
        F.binary_cross_entropy_with_logits(pos_sim, torch.ones_like(pos_sim))
        + F.binary_cross_entropy_with_logits(neg_sim, torch.zeros_like(neg_sim))
    ) / 2.0

    return alpha * attr_loss + (1.0 - alpha) * struct_loss


@torch.no_grad()
def dominant_anomaly_scores(
    model: DOMINANT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Per-node anomaly score = alpha * attr_score + (1-alpha) * struct_score.
    Both scores normalized to [0,1] before combining.
    Nodes with no edges get struct_score = 0 (only attr contributes).
    """
    model.eval()
    z, x_hat = model(x, edge_index)
    n = x.size(0)

    attr_scores = ((x - x_hat) ** 2).mean(dim=1)

    src, dst = edge_index[0], edge_index[1]
    sim = (z[src] * z[dst]).sum(dim=1)
    edge_err = F.binary_cross_entropy_with_logits(
        sim, torch.ones_like(sim), reduction="none"
    )
    struct_sum = torch.zeros(n, device=x.device)
    struct_cnt = torch.zeros(n, device=x.device)
    struct_sum.scatter_add_(0, src, edge_err)
    struct_sum.scatter_add_(0, dst, edge_err)
    struct_cnt.scatter_add_(0, src, torch.ones_like(edge_err))
    struct_cnt.scatter_add_(0, dst, torch.ones_like(edge_err))
    struct_scores = struct_sum / struct_cnt.clamp(min=1.0)

    def norm01(s: torch.Tensor) -> torch.Tensor:
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    combined = alpha * norm01(attr_scores) + (1.0 - alpha) * norm01(struct_scores)
    return combined.cpu().numpy()
