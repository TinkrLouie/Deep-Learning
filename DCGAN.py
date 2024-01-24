import os
import shutil
from cleanfid import fid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import numpy as np
import random
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torch.optim import Adam
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML

store_path = "vae_CIFAR100.pt"

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# hyperparameters
params = {
    'batch_size': 64,
    'nc': 3,
    'n_latent': 32,
    'lr': 0.002,
    'n_epochs': 60,
    'nz': 100,  # Size of z latent vector
    'real_label': 0.9,  # Label smoothing
    'fake_label': 0,
    'min_beta': 1e-4,
    'max_beta': 0.02,
    'beta1': 0.5,  # Hyperparameter for Adam
    'dim': 32,  # Image Size
    'ngf': 64,  # Size of feature maps for Generator
    'ndf': 64,  # Size of feature maps for Discriminator
    'num_workers': 2,
    'store_path': 'gan_model.pt'  # Store path for trained weights of model
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 3, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, i):
        return self.main(i)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 3, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, i):
        return self.main(i)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# nc = params['nc']
# nz = params['nz']
# ngf = params['ngf']
# ndf = params['ndf']
# lr = params['lr']
# beta1 = params['beta1']
# n_epochs = params['n_epochs']
# batch_size = params['batch_size']
# image_size = params['dim']
# real_label = params['real_label']
# fake_label = params['fake_label']

# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


if __name__ == '__main__':
    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    transform = Compose([
        # torchvision.transforms.Resize(40),
        # torchvision.transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ds = CIFAR100
    train_dataset = ds("./datasets", download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, params['batch_size'], shuffle=True)

    test_dataset = ds("./datasets", download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, params['batch_size'], shuffle=True)

    print(f'Size of training dataset: {len(train_loader.dataset)}')
    print(f'Size of testing dataset: {len(test_loader.dataset)}')

    netG = Generator(params['nc'], params['nz'], params['ngf']).to(device)
    netG.apply(weights_init)
    netD = Discriminator(params['nc'], params['ndf']).to(device)
    netD.apply(weights_init)
    criterion = nn.BCELoss().to(device)
    fixed_noise = torch.randn(params['batch_size'], params['nz'], 1, 1).to(device)
    optimizerD = Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

    total_params = len(torch.nn.utils.parameters_to_vector(netG.parameters())) + len(
        torch.nn.utils.parameters_to_vector(netD.parameters()))
    print(f'> Number of model parameters {total_params}')
    if total_params > 1000000:
        print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    n_samples = 10000
    real_images_dir = 'real_images'
    generated_images_dir = 'generated_images'

    print("Training:")

    for epoch in range(params['n_epochs']):
        for i, data in enumerate(train_loader, 0):
            if iters % 1000 == 1:
                print("Step: ", iters)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), params['real_label'], dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(params['fake_label'])
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(params['real_label'])  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, params['n_epochs'], i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == params['n_epochs'] - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Sample from Generator
            if epoch == params['n_epochs'] - 1 and i == len(train_loader) - 1:
                with torch.no_grad():
                    sample_noise = torch.randn(n_samples, params['nz'], 1, 1).to(device)
                    fake = netG(sample_noise).detach().cpu()

                # setup_directory(real_images_dir)
                setup_directory(generated_images_dir)

                for n, image in enumerate(fake):
                    save_image(image, os.path.join(generated_images_dir, f"gen_img_{n}.png"))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig('gen_img.png')

    # compute FID
    score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
    print(f"FID score: {score}")
