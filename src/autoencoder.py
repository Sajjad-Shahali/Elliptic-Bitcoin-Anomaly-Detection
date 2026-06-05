import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_train: np.ndarray,
    input_dim: int,
    latent_dim: int = 32,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> Autoencoder:
    model = Autoencoder(input_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            loss = criterion(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(X_train):.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model


def reconstruction_errors(model: Autoencoder, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    errors = []
    tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
            errors.append(mse)
    return np.concatenate(errors)


def threshold_predict(errors: np.ndarray, contamination: float = 0.065) -> np.ndarray:
    """Label top-contamination% errors as anomaly (1). Default matches test illicit rate."""
    cutoff = np.percentile(errors, 100 * (1 - contamination))
    return (errors >= cutoff).astype(int)
