import random
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from cleanfid import fid
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Lambda
import os
import shutil
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


# hyperparameters
params = {
    'batch_size': 64,
    'n_channels': 3,
    'n_latent': 32,
    'ngpu': 1,
    'lr': 0.001,
    'gen_lr': 0.0001,
    'dis_lr': 0.0004,
    'n_epoch': 50,
    'nz': 100,  # Size of z latent vector
    'real_label': 0.9,  # Label smoothing
    'fake_label': 0,
    'min_beta': 10 ** -4,
    'max_beta': 0.02,
    'dim': 32,
    'no_train': False
}


# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)
    ]
)

ds_fn = CIFAR100
train_dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
loader = DataLoader(train_dataset, params['batch_size'], shuffle=True)

test_dataset = ds_fn("./datasets", download=True, train=False, transform=transform)
test_loader = DataLoader(test_dataset, 10000, shuffle=True)

print(f'Size of training dataset: {len(loader.dataset)}')
print(f'Size of testing dataset: {len(test_loader.dataset)}')


class DDPM(nn.Module):
    def __init__(self, network, n_steps=1000, device=None):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(params['min_beta'], params['max_beta'], n_steps).to(
            device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        return self.network(x, t)

    def sample(self, latent, device=None):
        with torch.no_grad():
            if device is None:
                device = self.device

            # Copy latent to device
            latent = latent.to(device)

            # Start with the given latent
            x = latent

            for idx, t in enumerate(list(range(self.n_steps))[::-1]):
                # Estimate noise to be removed
                time_tensor = (torch.ones(latent.shape[0], 1) * t).to(device).long()
                eta_theta = self.backward(x, time_tensor)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                # Partially denoise the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn_like(latent)

                    # Choose sigma_t calculation method (adjust as needed)
                    beta_t = self.betas[t]
                    sigma_t = beta_t.sqrt()  # Or use beta_tilda_t if preferred

                    # Add controlled noise
                    x = x + sigma_t * z

        return x


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


class Block(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(Block, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 3)
        self.b1 = nn.Sequential(
            Block((3, 32, 32), 3, 10),
            Block((10, 32, 32), 10, 10),
            Block((10, 32, 32), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            Block((10, 16, 16), 10, 20),
            Block((20, 16, 16), 20, 20),
            Block((20, 16, 16), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            Block((20, 8, 8), 20, 40),
            Block((40, 8, 8), 40, 40),
            Block((40, 8, 8), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 5, 2, 2)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            Block((40, 4, 4), 40, 20),
            Block((20, 4, 4), 20, 20),
            Block((20, 4, 4), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 3, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            Block((80, 8, 8), 80, 40),
            Block((40, 8, 8), 40, 20),
            Block((20, 8, 8), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            Block((40, 16, 16), 40, 20),
            Block((20, 16, 16), 20, 10),
            Block((10, 16, 16), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            Block((20, 32, 32), 20, 10),
            Block((10, 32, 32), 10, 10),
            Block((10, 32, 32), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 3, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 4, 4)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))
        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))
        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    @staticmethod
    def _make_te(dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


def sample(ddpm, n_samples=10000, device=None, c=3, h=32, w=32):

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

    return x


store_path = "ddpm_CIFAR100.pt"

# Loading the trained model
best_model = DDPM(UNet(), n_steps=1000, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

# now show some interpolations (note you do not have to do linear interpolations as shown here, you can do non-linear or gradient-based interpolation if you wish)
z = sample(best_model, params['batch_size'], device=device)
col_size = int(np.sqrt(params['batch_size']))
# z[0:col_size] => Size([8, 3, 32, 32])
z0 = z[0:col_size].repeat(1, 1, col_size, 1)  # z for top row
z1 = z[params['batch_size'] - col_size:].repeat(1, 1, col_size, 1)  # z for bottom row
t = torch.linspace(0, 1, col_size).unsqueeze(1).repeat(1, col_size, 1, 1).to(device)
lerp_z = (1 - t) * z0 + t * z1  # linearly interpolate between two points in the latent space
lerp_g = best_model.sample(lerp_z, device=device)  # sample the model at the resulting interpolated latents
plt.rcParams['figure.dpi'] = 100
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(lerp_g).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
plt.show()


# define directories
#real_images_dir = 'real_images'
#generated_images_dir = 'generated_images'
#num_samples = 10000  # do not change
#
#
## create/clean the directories
#def setup_directory(directory):
#    if os.path.exists(directory):
#        shutil.rmtree(directory)  # remove any existing (old) data
#    os.makedirs(directory)
#
#
#setup_directory(real_images_dir)
#setup_directory(generated_images_dir)
#
## generate and save 10k model samples
#samples_batch = sample(best_model, num_samples, device=device)
#for i, image in enumerate(samples_batch):
#    save_image(image, os.path.join(generated_images_dir, f"gen_img_{i}.png"))
#
## save 10k images from the CIFAR-100 test dataset
#
#for batch in test_loader:
#    for i, image in enumerate(batch[0]):
#        save_image(image, os.path.join(real_images_dir, f"real_img_{i}.png"))
#
#
## compute FID
#score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
#print(f"FID score: {score}")