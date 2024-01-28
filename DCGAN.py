import os
import shutil
from cleanfid import fid
import torch
from torch import autograd
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
from torch.autograd import Variable
from torch.nn.utils import spectral_norm


# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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
    'lr': 0.0002,
    'steps': 50000,
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
    'store_path': 'dcgan_model.pt'  # Store path for trained weights of model
}

# Getting device ie. CPU or GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


# Helper functions to calculate shape of Tconv and conv output
def calculate_Tconv_output_size(input_size, padding, kernel_size, stride, output_padding=0, dilation=1):
    output = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    return output


def calculate_conv_output_size(input_size, padding, kernel_size, stride, dilation=1):
    output = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1

    return output


# TODO: Add Spectral Norm (Done) ->  Results : SN for both G&D = 101 | SN for D = 103 | 107 w/ GP
# TODO: Add Self-attention Layers (Done) -> Results = FID = 151 => Removed

# Reference: https://github.com/tcapelle/Diffusion-Models-pytorch/tree/main
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


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf, sn=False):
        super(Generator, self).__init__()

        self.nc = nc  # n channels
        self.nz = nz  # n latents
        self.ngf = ngf  # n features

        if sn:
            # Input Layer => [N, 128, 3, 3]
            self.input = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(nz, ngf * 2, 3, 1, 0, bias=False)),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )

            # Hidden Transposed Convolution Layer 1 => [N, 128, 5, 5]
            self.tconv1 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )

            # Hidden Transposed Convolution Layer 2 => [N, 128, 9, 9]
            self.tconv2 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )

            # Hidden Transposed Convolution Layer 3 => [N, 64, 17, 17]
            self.tconv3 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )

        else:
            # Input Layer => [N, 128, 3, 3]
            self.input = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 2, 3, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )

            # Hidden Transposed Convolution Layer 1 => [N, 128, 5, 5]
            self.tconv1 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )

            # Hidden Transposed Convolution Layer 2 => [N, 128, 9, 9]
            self.tconv2 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )

            # Hidden Transposed Convolution Layer 3 => [N, 64, 17, 17]
            self.tconv3 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )

        # Output Layer => [N, 3, 32, 32]
        self.output = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        output = self.output(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, sn=False):
        super(Discriminator, self).__init__()

        self.nc = nc  # n channels
        self.ndf = ndf  # n features

        if sn:
            # Input Layer => [N, 64, 17, 17]
            self.input = nn.Sequential(
                spectral_norm(nn.Conv2d(nc, ndf, 2, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Hidden Convolutional Layer 1 => [N, 128, 9, 9]
            self.conv1 = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Hidden Convolutional Layer 2 => [N, 128, 5, 5]
            self.conv2 = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Hidden Convolutional Layer 3 => [N, 256, 3, 3]
            self.conv3 = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True)
            )

        else:
            # Input Layer => [N, 64, 17, 17]
            self.input = nn.Sequential(
                nn.Conv2d(nc, ndf, 2, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Hidden Convolutional Layer 1 => [N, 128, 9, 9]
            self.conv1 = nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Hidden Convolutional Layer 2 => [N, 128, 5, 5]
            self.conv2 = nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 3, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 3),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Hidden Convolutional Layer 3 => [N, 256, 3, 3]
            self.conv3 = nn.Sequential(
                nn.Conv2d(ndf * 3, ndf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Output Layer => [N, 1, 1, 1]
        self.output = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.output(x)

        return output


# Weight function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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

    # Train iterable
    train_iterator = iter(cycle(train_loader))

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
    bce = nn.BCELoss().to(device)


    # ---------------------
    # Initialise optimiser
    # ---------------------
    optimizerD = Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

    # Check how many parameters in total between 2 models
    total_params = len(torch.nn.utils.parameters_to_vector(netG.parameters())) + len(
        torch.nn.utils.parameters_to_vector(netD.parameters()))
    print(f'> Number of model parameters {total_params}')
    if total_params > 1000000:
        print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Scalar tensor for loss scaling in WGAN-GP
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = (one * -1).to(device)

    # Directory names for image storage
    n_samples = 10000
    real_images_dir = 'real_images'
    generated_images_dir = 'generated_images'

    print("Training:")

    # ---------
    # Training
    # ---------
    while iters < params['steps']:
        for i, data in enumerate(train_loader, 0):
            # ---------------------------
            # Update Discriminator Model
            # ---------------------------

            # Train with real images
            netD.zero_grad()
            data = data[0].to(device)
            b_size = data.size(0)
            real_label = torch.full((b_size,), params['real_label'], dtype=torch.float, device=device)
            fake_label = torch.full((b_size,), params['fake_label'], dtype=torch.float, device=device)
            # Forward pass
            output = netD(data).view(-1)
            # Loss of real images
            errD_real = bce(output, real_label)
            #errD_real = output.mean()
            # Gradients
            errD_real.backward()

            # Train with fake images
            # Generate latent vectors with batch size indicated in params
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake = netG(noise)

            # Classify fake images with Discriminator
            output = netD(fake.detach()).view(-1)
            # Discriminator's loss on the fake images
            errD_fake = bce(output, fake)
            #errD_fake = output.mean()
            # Gradients for backward pass
            errD_fake.backward()

            # TODO: GP function (Done) -> Results = FID = 115 | 107 w/ SN
            #gp = gradient_penalty(netD, data, fake.detach())
            #gp.backward()
            # Compute sum error of Discriminator
            errD = errD_fake + errD_real
            #errD = errD_fake - errD_real + gp
            # Update Discriminator
            optimizerD.step()

            # -----------------------
            # Update Generator Model
            # -----------------------

            netG.zero_grad()
            # Forward pass of fake images through Discriminator
            output = netD(fake).view(-1)
            # G's loss based on this output
            errG = bce(output, real_label)
            #errG = output.mean()
            # Calculate gradients for Generator
            errG.backward()
            # Update Generator
            optimizerG.step()

            # Output training stats
            if iters % 1000 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (iters, params['steps'], errD.item(), errG.item()))

            # Save Losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Sample for visualisation
            if iters == params['steps'] - 1:
                fixed_noise = torch.randn(params['batch_size'], params['nz'], 1, 1).to(device)
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # ---------------------------------------------------------
    # Sampling from latent space and save 10000 samples to dir
    # ---------------------------------------------------------
    setup_directory(generated_images_dir)
    n = 0
    sample_noise = torch.randn(1000, params['nz'], 1, 1).to(device)
    for i in range(10):
        with torch.no_grad():
            fake = netG(sample_noise).detach().cpu()

        for image in fake:
            save_image(image, os.path.join(generated_images_dir, f"gen_img_{n}.png"))
            n += 1


    # ---------------------
    # Linear Interpolation
    # ---------------------
    sample_noise = torch.randn(params['batch_size'], params['nz'], 1, 1).to(device)
    col_size = int(np.sqrt(params['batch_size']))

    z0 = sample_noise[0:col_size].repeat(col_size, 1, 1, 1)  # z for top row
    z1 = sample_noise[params['batch_size'] - col_size:].repeat(col_size, 1, 1, 1)  # z for bottom row

    t = torch.linspace(0, 1, col_size).unsqueeze(1).repeat(1, col_size).view(params['batch_size'], 1, 1, 1).to(device)
    lerp_z = (1 - t) * z0 + t * z1  # linearly interpolate between two points in the latent space
    with torch.no_grad():
        lerp_g = netG(lerp_z)  # sample the model at the resulting interpolated latents

    print(f'Discriminator statistics: mean = {np.average(D_losses)}, stdev = {np.std(D_losses)},')
    plt.figure(figsize=(10, 5))
    plt.title('Interpolation')
    plt.rcParams['figure.dpi'] = 100
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(lerp_g).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
    plt.savefig('interpolation.png')

    # ------------------------------
    # Plot figures using matplotlib
    # ------------------------------
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

    # ------------
    # compute FID
    # ------------
    score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
    print(f"FID score: {score}")
