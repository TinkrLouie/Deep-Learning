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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F
from torch.autograd import Variable

store_path = "vae_CIFAR100.pt"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


class VAE(nn.Module):

    def __init__(self, image_size, hidden_dim, encoding_dim):
        super(VAE, self).__init__()

        self.encoding_dim = encoding_dim
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(256, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 256)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        # Obtain mu and logvar
        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)

        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Decode
        decoded = self.decode(z)

        # return decoded, mu, logvar
        return decoded, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


model = VAE(32, 100, 20)
optimizer = Adam(model.parameters(), lr=1e-3)


#Train model
def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


real_images_dir = 'real_images'
generated_images_dir = 'VAE_results'
setup_directory(generated_images_dir)

num_epochs = 30
for epoch in range(1, num_epochs):
    train(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20)
        sample = model.decode(sample)
        save_image(sample.view(64, 3, 32, 32), 'VAE_results/sample_' + str(epoch) + '.png')


# compute FID
score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
print(f"FID score: {score}")