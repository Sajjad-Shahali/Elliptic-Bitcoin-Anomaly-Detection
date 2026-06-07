import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops


class GraphSAGE(nn.Module):
    """Original 2-layer GraphSAGE — kept for backward compat."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GraphSAGEv2(nn.Module):
    """
    Improved 3-layer GraphSAGE:
    - 3 conv layers (captures 3-hop neighborhoods)
    - LayerNorm after each layer
    - Self-loops added in forward pass
    - Larger hidden dim recommended (256)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim,  hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        ei, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.norm1(F.relu(self.conv1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm2(F.relu(self.conv2(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm3(F.relu(self.conv3(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward(); optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    out    = model(data.x.to(device), data.edge_index.to(device))
    probs  = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    preds  = out.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()
    return preds[mask], probs[mask], labels[mask]


def class_weights(y_train_tensor, device):
    counts  = torch.bincount(y_train_tensor)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()
    return weights.to(device)
