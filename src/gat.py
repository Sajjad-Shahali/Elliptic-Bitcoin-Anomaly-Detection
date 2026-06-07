import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Original 2-layer GAT — kept for backward compat."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)


class GATv2(nn.Module):
    """
    Improved 3-layer GAT:
    - heads=2 (was 4) — prevents param explosion
    - Residual projection on every layer
    - LayerNorm for stable training
    - Higher dropout (set at call time)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2,
                 heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim,         hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.conv2 = GATConv(hidden_dim,         hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.conv3 = GATConv(hidden_dim,         hidden_dim, heads=1,     dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # residual projections (input_dim != hidden_dim)
        self.res1 = nn.Linear(input_dim,  hidden_dim, bias=False)
        self.res2 = nn.Identity()
        self.res3 = nn.Identity()
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm1(F.elu(self.conv1(x, edge_index)) + self.res1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm2(F.elu(self.conv2(x, edge_index)) + self.res2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm3(F.elu(self.conv3(x, edge_index)) + self.res3(x))
        return self.classifier(x)


def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward(); optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    out    = model(data.x, data.edge_index)
    probs  = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    preds  = out.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()
    return preds[mask], probs[mask], labels[mask]
