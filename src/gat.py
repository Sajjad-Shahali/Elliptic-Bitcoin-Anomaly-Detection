import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim,  hidden_dim, heads=heads,     dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
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
