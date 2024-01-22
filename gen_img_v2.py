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
from collections import OrderedDict

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
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


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
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        #self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(64, 128)
        #self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(128, 128)
        #self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(128, 256)
        #self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        #self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(128, 32)
        #self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(64, 64)
        #self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        #x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        #x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        #x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        #x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        #x = self.sa4(x)
        x = self.up2(x, x2, t)
        #x = self.sa5(x)
        x = self.up3(x, x1, t)
        #x = self.sa6(x)
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


# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    transform = Compose([
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

    store_path = "ddpm_CIFAR100.pt"

    # Loading the trained model
    best_model = UNet().to(device)
    state_dict = torch.load(store_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[8:]
        new_state_dict[name] = v
    best_model.load_state_dict(new_state_dict)
    print("Model loaded")
    diffusion = Diffusion(device=device)

    real_images_dir = 'real_images'
    generated_images_dir = 'generated_images'
    num_samples = 10000  # do not change

    #setup_directory(real_images_dir)
    setup_directory(generated_images_dir)

    # generate and save 10k model samples
    samples = diffusion.sample(best_model, 10)
    for i, image in enumerate(samples):
        save_image(image, os.path.join(generated_images_dir, f"gen_img_{i}.png"))

    # compute FID
    score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
    print(f"FID score: {score}")