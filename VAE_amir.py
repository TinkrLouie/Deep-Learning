import os
import shutil
from cleanfid import fid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

store_path = "vae_CIFAR100.pt"

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = 64
n_channels = 3
n_latent = 64
lr = 0.001
n_epoch = 50
n_steps = 1000
min_beta = 1e-4
max_beta = 0.02
dim = 32
num_classes = 100


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
transform = Compose([
    # torchvision.transforms.Resize(40),
    # torchvision.transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
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


# define the model
class VAE(nn.Module):
    def __init__(self, n_channels=1, f_dim=32 * 20 * 20, z_dim=256):
        super().__init__()

        # encoder layers:
        self.enc_conv1 = nn.Conv2d(n_channels, 16, 5)
        self.enc_conv2 = nn.Conv2d(16, 32, 5)
        # two linear layers with one for the mean and the other the variance
        self.enc_linear1 = nn.Linear(f_dim, z_dim)
        self.enc_linear2 = nn.Linear(f_dim, z_dim)

        # decoder layers:
        self.dec_linear = nn.Linear(z_dim, f_dim)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, 5)
        self.dec_conv2 = nn.ConvTranspose2d(16, n_channels, 5)

    # encoder:
    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(-1, 32 * 20 * 20)
        # the output is mean (mu) and variance (logVar)
        mu = self.enc_linear1(x)
        logVar = self.enc_linear2(x)
        # mu and logVar are used to sample z and compute KL divergence loss
        return mu, logVar

    # reparameterisation trick:
    def reparameterise(self, mu, logVar):
        # from mu and logVar, we can sample via mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    # decoder:
    def decoder(self, z):
        x = F.relu(self.dec_linear(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.dec_conv1(x))
        # the output is the same size as the input
        x = torch.sigmoid(self.dec_conv2(x))
        return x

    # forward pass:
    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterise(mu, logVar)
        out = self.decoder(z)
        # mu and logVar are returned as well as the output for loss computation
        return out, mu, logVar


model = VAE().to(device)
print(f'The model has {len(torch.nn.utils.parameters_to_vector(model.parameters()))} parameters.')

print('The optimiser has been created!')
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)


# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


real_images_dir = 'real_images'
generated_images_dir = 'VAE_results'
setup_directory(generated_images_dir)


epoch = 0
# training loop
while epoch < 20:

    # for metrics
    loss_arr = np.zeros(0)

    # iterate over the training dateset
    for i, batch in enumerate(train_loader):
        # sample x from the dataset
        x, _ = batch
        x = x.to(device)

        # forward pass to obtain image, mu, and logVar
        x_hat, mu, logVar = model(x)

        # caculate loss - BCE combined with KL
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(x_hat, x, size_average=False) + kl_divergence

        # backpropagate to compute the gradients of the loss w.r.t the parameters and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # collect stats
        loss_arr = np.append(loss_arr, loss.item())

    # sample
    z = torch.randn_like(mu)
    print(z.shape)
    g = model.decoder(z)
    print(g.shape)

    save_image(g.view(64, 3, 32, 32), 'VAE_results/sample_' + str(epoch) + '.png')

    epoch += 1

# compute FID
score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
print(f"FID score: {score}")