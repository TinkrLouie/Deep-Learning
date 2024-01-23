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


# define the generator
class Generator(nn.Module):
    def __init__(self, latent_size=100, label_size=10):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size + label_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, n_channels * dim * dim),
            nn.Tanh()
        )

    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        x = torch.cat((x, c), 1)  # [input, label] concatenated
        x = self.layer(x)
        return x.view(x.size(0), n_channels, dim, dim)


# define the discriminator
class Discriminator(nn.Module):
    def __init__(self, label_size=10):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_channels * dim * dim + label_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        x = torch.cat((x, c), 1)  # [input, label] concatenated
        return self.layer(x)


G = Generator().to(device)
D = Discriminator().to(device)

print(f'Generator has {len(torch.nn.utils.parameters_to_vector(G.parameters()))} parameters.')
print(f'Discriminator has {len(torch.nn.utils.parameters_to_vector(D.parameters()))} parameters')

# initialise the optimiser
optimiser_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimiser_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
print('Optimisers have been created!')

criterion = nn.BCELoss()
epoch = 0
print('Loss function is Binary Cross Entropy!')


# training loop
while epoch < 20000:

    # arrays for metrics
    logs = {}
    gen_loss_arr = np.zeros(0)
    dis_loss_arr = np.zeros(0)

    # iterate over the train dateset
    for i, batch in enumerate(train_loader):
        x, t = batch
        x, t = x.to(device), t.to(device)

        # convert target labels "t" to a one-hot vector, e.g. 3 becomes [0,0,0,1,0,0,0,...]
        y = torch.zeros(x.size(0), 10).long().to(device).scatter(1, t.view(x.size(0), 1), 1)

        # train discriminator
        z = torch.randn(x.size(0), 100).to(device)
        l_r = criterion(D(x, y), torch.ones([64, 1]).to(device))  # real -> 1
        l_f = criterion(D(G(z, y), y), torch.zeros([64, 1]).to(device))  # fake -> 0
        loss_d = (l_r + l_f) / 2.0
        optimiser_D.zero_grad()
        loss_d.backward()
        optimiser_D.step()

        # train generator
        z = torch.randn(x.size(0), 100).to(device)
        loss_g = criterion(D(G(z, y), y), torch.ones([64, 1]).to(device))  # fake -> 1
        optimiser_G.zero_grad()
        loss_g.backward()
        optimiser_G.step()

        gen_loss_arr = np.append(gen_loss_arr, loss_g.item())
        dis_loss_arr = np.append(dis_loss_arr, loss_d.item())

    # conditional sample of 10x10
    G.eval()
    print('loss d: {:.3f}, loss g: {:.3f}'.format(gen_loss_arr.mean(), dis_loss_arr.mean()))
    grid = np.zeros([dim * 10, dim * 10])
    for j in range(10):
        c = torch.zeros([10, 10]).to(device)
        c[:, j] = 1
        z = torch.randn(10, 100).to(device)
        y_hat = G(z, c).view(10, dim, dim)
        result = y_hat.cpu().data.numpy()
        grid[j * dim:(j + 1) * dim] = np.concatenate([x for x in result], axis=-1)
    plt.grid(False)
    plt.imshow(grid, cmap='gray')
    plt.show()
    plt.pause(0.0001)
    G.train()

    epoch = epoch + 1