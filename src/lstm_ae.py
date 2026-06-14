import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (batch, hidden_dim) — last-layer final hidden state


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, output_dim: int = 165, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch = z.size(0)
        inp = z.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
        h0  = z.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0  = torch.zeros_like(h0)
        out, _ = self.lstm(inp, (h0, c0))
        return self.proj(out)  # (batch, T, output_dim)


class LSTMAE(nn.Module):
    """
    LSTM Autoencoder for temporal anomaly detection on per-timestep aggregate features.
    Encoder: LSTM → bottleneck h_T.
    Decoder: LSTM conditioned on h_T, reconstructs T-step sequence.
    """
    def __init__(self, input_dim: int = 165, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_dim, input_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor):
        z   = self.encoder(x)
        out = self.decoder(z, x.size(1))
        return out, z


def build_timestep_sequences(df, feature_cols: list, window_size: int = 10):
    """
    1. Compute per-timestep mean feature vector across all transactions at that step.
    2. Extract sliding windows of size `window_size`.

    Returns:
        windows    - float32 array (num_windows, window_size, n_features)
        step_means - float32 array (n_steps, n_features)
        step_ids   - list of time step ids in order
    """
    step_ids   = sorted(df["time_step"].unique())
    step_means = np.stack([
        df[df["time_step"] == t][feature_cols].values.mean(axis=0).astype(np.float32)
        for t in step_ids
    ], axis=0)

    windows = np.stack([
        step_means[i : i + window_size]
        for i in range(len(step_ids) - window_size + 1)
    ], axis=0)

    return windows, step_means, step_ids


def train_lstm_ae(
    windows: np.ndarray,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> LSTMAE:
    model = LSTMAE(input_dim, hidden_dim, num_layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X      = torch.tensor(windows, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(batch)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total/len(windows):.6f}  lr={lr_now:.2e}")
    return model


@torch.no_grad()
def score_timesteps(model: LSTMAE, step_means: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Per-timestep anomaly score = mean reconstruction MSE over all windows containing that step.
    (Course Theorem 4.1: each interior step covered by exactly T windows → smoothing effect.)
    """
    model.eval()
    n_steps     = len(step_means)
    step_scores = np.zeros(n_steps)
    step_counts = np.zeros(n_steps)

    X = torch.tensor(step_means, dtype=torch.float32)
    for i in range(n_steps - window_size + 1):
        window = X[i : i + window_size].unsqueeze(0).to(DEVICE)
        recon, _ = model(window)
        per_step_err = ((recon.squeeze(0) - window.squeeze(0)) ** 2).mean(dim=1).cpu().numpy()
        step_scores[i : i + window_size] += per_step_err
        step_counts[i : i + window_size] += 1.0

    return step_scores / np.maximum(step_counts, 1.0)
