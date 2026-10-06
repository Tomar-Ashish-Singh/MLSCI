import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "image_size": 32,
    "channels": 1,
    "batch_size": 32,
    "epochs": 120,
    "lr": 2e-3,
    "timesteps": 200,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "dataset_size": 800,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# Synthetic Lens Simulation
# ==========================================
def make_mock_lenses(n_samples=800, size=32):
    """Generates synthetic 2D lensing images."""
    grid = torch.linspace(-1.0, 1.0, size)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    images = torch.zeros((n_samples, 1, size, size), dtype=torch.float32)

    for i in range(n_samples):
        # Deflector galaxy profile
        core_scale = np.random.uniform(0.8, 1.2)
        core = core_scale * torch.exp(-35.0 * (r**2))

        # Perturbed Einstein ring
        eps = np.random.uniform(0.0, 0.15)
        r_dist = r * (1.0 + eps * torch.cos(2 * theta))
        r_ring = np.random.uniform(0.45, 0.65)
        w_ring = np.random.uniform(60.0, 90.0)
        ring = torch.exp(-w_ring * ((r_dist - r_ring) ** 2))

        # Sensor background noise
        noise = 0.04 * torch.randn(size, size)

        sample = core + ring + noise
        sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
        images[i, 0] = sample * 2.0 - 1.0  # normalize to [-1, 1]

    return images

# ==========================================
# DDPM Variance Schedule
# ==========================================
class DDPM:
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)
        alpha_hat_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_hat[:-1]])
        self.posterior_var = self.betas * (1.0 - alpha_hat_prev) / (1.0 - self.alpha_hat)

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        s_alpha = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        s_one_minus_alpha = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        return s_alpha * x0 + s_one_minus_alpha * noise

# ==========================================
# U-Net Architecture
# ==========================================
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -factor)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.act = nn.SiLU()
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x, t):
        h = self.act(self.conv1(x))
        t_emb = self.act(self.time_proj(t))
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)
        return self.norm(self.act(self.conv2(h)))

class LensUNet(nn.Module):
    def __init__(self, in_channels=1, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.down1 = ResidualBlock(in_channels, 32, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResidualBlock(32, 64, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = ResidualBlock(64, 128, time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = ResidualBlock(128, 64, time_dim)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = ResidualBlock(64, 32, time_dim)
        self.out = nn.Conv2d(32, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        m = self.mid(self.pool2(x2), t_emb)
        u1 = self.conv_up1(torch.cat([self.up1(m), x2], dim=1), t_emb)
        u2 = self.conv_up2(torch.cat([self.up2(u1), x1], dim=1), t_emb)
        return self.out(u2)

# ==========================================
# Sampling Function
# ==========================================
@torch.no_grad()
def p_sample(model, ddpm, num_samples=4, img_size=32, device="cpu"):
    model.eval()
    x = torch.randn((num_samples, 1, img_size, img_size), device=device)
    for i in reversed(range(ddpm.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        mean = (1.0 / torch.sqrt(ddpm.alphas[i])) * (x - (ddpm.betas[i] / torch.sqrt(1.0 - ddpm.alpha_hat[i])) * eps_pred)
        if i > 0: x = mean + torch.sqrt(ddpm.posterior_var[i]) * torch.randn_like(x)
        else: x = mean
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0

# ==========================================
# Main Execution (Train & Evaluate)
# ==========================================
if __name__ == "__main__":
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    dataset = make_mock_lenses(CONFIG["dataset_size"], CONFIG["image_size"])
    loader = DataLoader(TensorDataset(dataset), batch_size=CONFIG["batch_size"], shuffle=True)

    ddpm = DDPM(timesteps=CONFIG["timesteps"], beta_start=CONFIG["beta_start"], beta_end=CONFIG["beta_end"], device=device)
    model = LensUNet(in_channels=CONFIG["channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, CONFIG["epochs"] + 1):
        running_loss = 0.0
        for (batch,) in loader:
            x0 = batch.to(device)
            t = torch.randint(0, CONFIG["timesteps"], (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)
            pred_noise = model(ddpm.q_sample(x0, t, noise), t)
            loss = criterion(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{CONFIG['epochs']} | MSE Loss: {running_loss / len(loader):.5f}")

    print("Generating evaluation samples...")
    sampled_lenses = p_sample(model, ddpm, num_samples=4, img_size=CONFIG["image_size"], device=device)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axes[0, i].imshow((dataset[i, 0].numpy() + 1.0) / 2.0, cmap="inferno")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Target #{i+1}", fontsize=9)
        axes[1, i].imshow(sampled_lenses[i, 0].cpu().numpy(), cmap="inferno")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Sample #{i+1}", fontsize=9)
    plt.tight_layout()
    plt.savefig("results.png", dpi=200)
    plt.show()

    torch.save(model.state_dict(), "astro_diffusion_model.pt")
    print("Saved weights to astro_diffusion_model.pt and comparison to results.png")
