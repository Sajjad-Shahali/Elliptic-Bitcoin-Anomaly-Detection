import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64),        nn.BatchNorm1d(64),         nn.ReLU(),
        )
        self.mu_layer      = nn.Linear(64, latent_dim)
        self.logvar_layer  = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),   nn.BatchNorm1d(64),         nn.ReLU(),
            nn.Linear(64, hidden_dim),   nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl_loss) / x.size(0)


def train_vae(
    X_train: np.ndarray,
    input_dim: int,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    beta: float = 1.0,
) -> VAE:
    model = VAE(input_dim, hidden_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(batch)
        scheduler.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total/len(X_train):.4f}")

    return model


@torch.no_grad()
def vae_anomaly_scores(model: VAE, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size
    )
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        recon, mu, logvar = model(batch)
        mse = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
        scores.append(mse)
    return np.concatenate(scores)
