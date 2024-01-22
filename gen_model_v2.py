import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from cleanfid import fid
from torchvision.utils import save_image
import os
from torchvision.datasets import CIFAR100
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F

store_path = "ddpm_CIFAR100.pt"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = 64
n_channels = 3
n_latent = 64
lr = 0.0002
n_epoch = 50
n_steps = 1000
min_beta = 1e-4
max_beta = 0.02
dim = 32
num_classes = 100


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32)
        self.sa1 = SelfAttention(32)
        self.down2 = Down(32, 32)
        self.sa2 = SelfAttention(32)
        self.down3 = Down(32, 64)
        self.sa3 = SelfAttention(64)

        self.bot1 = DoubleConv(64, 64)
        self.bot3 = DoubleConv(64, 64)

        self.up1 = Up(128, 32)
        self.sa4 = SelfAttention(32)
        self.up2 = Up(64, 32)
        self.sa5 = SelfAttention(32)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    @staticmethod
    def pos_encoding(t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample(self, model, num_samples=10000):
        model.eval()
        with torch.no_grad():
            x = torch.randn((num_samples, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(num_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(dataloader, n_epochs=50, store_path="ddpm_model.pt"):
    best_loss = float("inf")
    model = UNet().to(device)

    total_params = len(nn.utils.parameters_to_vector(model.parameters()))
    print(f'> Number of model parameters {total_params}')
    if total_params > 1000000:
        print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

    optimizer = AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(device=device)
    l = len(dataloader)
    # TODO: time each epoch
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = torch.randint(low=1, high=diffusion.noise_steps, size=(images.shape[0],)).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(images) / l

        # Display images generated at this epoch
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), store_path)
            log_string += " --> Current best model (stored)"

        print(log_string)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    transform = Compose([
        torchvision.transforms.Resize(40),
        torchvision.transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ds = CIFAR100
    train_dataset = ds("./datasets", download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = ds("./datasets", download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    print(f'Size of training dataset: {len(train_loader.dataset)}')
    print(f'Size of testing dataset: {len(test_loader.dataset)}')

    train(train_loader)