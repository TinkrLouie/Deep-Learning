import os
import shutil
from cleanfid import fid
import torch
import torch.nn as nn
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
from torch.nn.utils.parametrizations import spectral_norm
from torch.autograd import Variable
from torch import autograd


# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# hyperparameters
params = {
    'batch_size': 64,
    'nc': 3,
    'lr': 0.002,
    'step': 50000,
    'nz': 100,  # Size of z latent vector
    'real_label': 0.9,  # Label smoothing
    'fake_label': 0,
    'beta1': 0.5,  # Hyperparameter for Adam
    'dim': 32,  # Image Size
    'ngf': 64,  # Size of feature maps for Generator
    'ndf': 64,  # Size of feature maps for Discriminator
    'store_path': 'gan_model.pt'  # Store path for trained weights of model
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

# TODO: Add Spectral Norm (Done) ->  Results : SN for both G&D = 87 | SN for D = ?
# TODO: Add Self-attention Layers (Done) -> Results = FID = 151 => Removed


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
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 2, bias=False),
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


# Gradient Penalty function for WGAN-GP
# Reference: https://github.com/Zeleni9/pytorch-wgan/tree/master
def gradient_penalty(D, real_images, fake_images, lambda_term=10):
    eta = torch.FloatTensor(real_images.size(0), 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(real_images.size(0), 3, 32, 32).to(device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    # flatten the gradients to it calculates norm batchwise
    gradients = gradients.view(gradients.size(0), -1)

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty


# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    # ---------------------------------------------------------------------------------------
    transform = Compose([
        # torchvision.transforms.Resize(40),
        # torchvision.transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ds = CIFAR100

    # Create train batch loader
    train_dataset = ds("./datasets", download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, params['batch_size'], shuffle=True)

    # Create test batch loader
    test_dataset = ds("./datasets", download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, params['batch_size'], shuffle=True)

    print(f'Size of training dataset: {len(train_loader.dataset)}')
    print(f'Size of testing dataset: {len(test_loader.dataset)}')

    # ------------------------------------
    # Initialise Models and apply weights
    # ------------------------------------
    netG = Generator(params['nc'], params['nz'], params['ngf']).to(device)
    netG.apply(weights_init)
    netD = Discriminator(params['nc'], params['ndf']).to(device)
    netD.apply(weights_init)


    # --------------
    # Loss function
    # --------------
    criterion = nn.BCELoss().to(device)


    # ---------------------
    # Initialise optimiser
    # ---------------------
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

    # Scalar tensor for loss scaling in WGAN-GP
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = (one * -1).to(device)


    print("Training:")

    while iters < params['step']:
        for i, data in enumerate(train_loader, 0):
            if iters >= params['steps']:
                break
            # TODO: Implement CGAN
            # ---------------------------
            # Update Discriminator Model
            # ---------------------------

            # Train with real images
            netD.zero_grad()
            data = data[0].to(device)
            b_size = data.size(0)
            # Use one-sided label smoothing where real labels are filled with 0.9 instead of 1
            label = torch.full((b_size,), params['real_label'], dtype=torch.float, device=device)
            # Forward pass
            output = netD(data).view(-1)
            # Loss of real images
            #errD_real = criterion(output, label)
            errD_real = output.mean()
            # Gradients
            errD_real.backward(mone)

            # Train with fake images
            # Generate latent vectors with batch size indicated in params
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake = netG(noise)

            # One-sided label smoothing where fake labels are filled with 0
            label.fill_(params['fake_label'])
            # Classify fake images with Discriminator
            output = netD(fake.detach()).view(-1)
            # Discriminator's loss on the fake images
            #errD_fake = criterion(output, label)
            errD_fake = output.mean()
            # Gradients for backward pass
            errD_fake.backward()

            # TODO: GP function (Done) -> Results = FID = ?
            # gp = gradient_penalty(netD, data, fake.detach())
            # gp.backward()
            # Compute sum error of Discriminator
            errD = errD_fake + errD_real
            # errD = errD_fake - errD_real + gp
            # Update Discriminator
            optimizerD.step()

            # -----------------------
            # Update Generator Model
            # -----------------------

            netG.zero_grad()
            label.fill_(params['real_label'])  # fake labels are real for generator cost
            # Forward pass of fake images through Discriminator
            output = netD(fake).view(-1)
            # G's loss based on this output
            #errG = criterion(output, label)
            errG = output.mean()
            # Calculate gradients for Generator
            errG.backward(mone)
            # Update Generator
            optimizerG.step()

            # Output training stats
            if iters % 1000 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (iters, params['step'], i, len(train_loader),
                         errD.item(), errG.item()))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

    with torch.no_grad():
        sample_noise = torch.randn(n_samples, params['nz'], 1, 1).to(device)
        fake = netG(sample_noise).detach().cpu()

    setup_directory(generated_images_dir)

    for n, image in enumerate(fake):
        save_image(image, os.path.join(generated_images_dir, f"gen_img_{n}.png"))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig('gen_img.png')

    # compute FID
    score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
    print(f"FID score: {score}")
