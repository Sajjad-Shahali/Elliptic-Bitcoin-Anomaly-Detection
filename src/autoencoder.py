import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from scipy.stats import genpareto

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    """Original AE — kept for backward compat."""
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


class DenoisingAutoencoder(nn.Module):
    """
    Improved AE:
    - Smaller latent dim (8) — tighter bottleneck, harder to reconstruct outliers
    - Deeper encoder 166->256->128->64->8
    - Denoising: Gaussian noise added to input during training
    - noise_std controls corruption level
    """
    def __init__(self, input_dim: int, latent_dim: int = 8, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
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
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            loss = criterion(model(batch), batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(X_train):.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
    return model


def train_denoising_autoencoder(
    X_train: np.ndarray,
    input_dim: int,
    latent_dim: int = 8,
    noise_std: float = 0.1,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> DenoisingAutoencoder:
    model = DenoisingAutoencoder(input_dim, latent_dim, noise_std).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    # clean inputs as targets, noisy as input (handled inside forward)
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            clean = batch.to(DEVICE)
            recon = model(clean)
            loss  = criterion(recon, clean)   # target = clean original
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(X_train):.6f}")
    return model


def reconstruction_errors(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    errors = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            out   = model(batch)
            # DenoisingAE: disable noise in eval (training=False already)
            mse   = ((out - batch) ** 2).mean(dim=1).cpu().numpy()
            errors.append(mse)
    return np.concatenate(errors)


def threshold_predict(errors: np.ndarray, contamination: float = 0.065) -> np.ndarray:
    cutoff = np.percentile(errors, 100 * (1 - contamination))
    return (errors >= cutoff).astype(int)


def f1_optimal_threshold(scores: np.ndarray, y_val: np.ndarray) -> float:
    """Find threshold on validation set that maximises F1 on illicit class."""
    best_f1, best_t = 0.0, 0.5
    for t in np.percentile(scores, np.linspace(50, 99, 100)):
        preds = (scores >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def evt_threshold(scores: np.ndarray, tail_quantile: float = 0.90, exceedance_prob: float = 0.01) -> float:
    """
    Extreme Value Theory threshold via Generalised Pareto Distribution (GPD).
    Fits GPD to the upper tail of `scores` above `tail_quantile`-percentile.
    Returns threshold at which P(score > t | score > u) = exceedance_prob.

    Course reference: Module 4, threshold selection §5.1(iii).
    More principled than fixed-percentile for heavy-tailed reconstruction errors.
    """
    u = np.percentile(scores, 100 * tail_quantile)
    exceedances = scores[scores > u] - u
    if len(exceedances) < 10:
        # Fallback to simple percentile if not enough tail samples
        return np.percentile(scores, 100 * (1 - exceedance_prob))
    c, loc, scale = genpareto.fit(exceedances, floc=0)
    # P(X > t | X > u) = exceedance_prob  →  solve for t
    t_above_u = genpareto.ppf(1.0 - exceedance_prob, c, loc=loc, scale=scale)
    return u + t_above_u
