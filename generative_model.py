import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from cleanfid import fid
import torchvision.utils as vutils
from torchvision.utils import save_image
import os
from torchvision.datasets import CIFAR100
import random
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# hyperparameters
params = {
    'batch_size': 64,
    'n_channels': 3,
    'n_latent': 32,
    'ngpu': 1,
    'lr': 0.001,
    'gen_lr': 0.0001,
    'dis_lr': 0.0004,
    'n_epoch': 50000,
    'nz': 100,  # Size of z latent vector
    'real_label': 0.9,  # Label smoothing
    'fake_label': 0,
    'min_beta': 10 ** -4,
    'max_beta': 0.02,
    'dim': 32,
    'no_train': False
}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# helper function to make getting another batch of data easier
#def cycle(iterable):
#    while True:
#        for x in iterable:
#            yield x


class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
               'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
               'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
               'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard',
               'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
               'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
               'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
               'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
               'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
               'willow_tree', 'wolf', 'woman', 'worm', ]

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)

#train_loader = DataLoader(
#    torchvision.datasets.CIFAR100('datasets', train=True, download=True, transform=torchvision.transforms.Compose([
#        torchvision.transforms.ToTensor()
#    ])),
#    batch_size=64, drop_last=True)
#
#test_loader = DataLoader(
#    torchvision.datasets.CIFAR100('datasets', train=False, download=True, transform=torchvision.transforms.Compose([
#        torchvision.transforms.ToTensor()
#    ])),
#    batch_size=64, drop_last=True)

ds_fn = CIFAR100
dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
loader = DataLoader(dataset, params['batch_size'], shuffle=True)


# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


print(f'Size of dataset: {len(loader.dataset)}')
#train_iterator = iter(cycle(train_loader))
#test_iterator = iter(cycle(test_loader))
#
#print(f'> Size of training dataset {len(train_loader.dataset)}')
#print(f'> Size of test dataset {len(test_loader.dataset)}')


# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=1000, device=None, image_chw=(params['n_channels'], params['dim'], params['dim'])):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(params['min_beta'], params['max_beta'], n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
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


class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 3)
        self.b1 = nn.Sequential(
            MyBlock((params['n_channels'], params['dim'], params['dim']), 3, 10),
            MyBlock((10, 32, 32), 10, 10),
            MyBlock((10, 32, 32), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 16, 16), 10, 20),
            MyBlock((20, 16, 16), 20, 20),
            MyBlock((20, 16, 16), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 8, 8), 20, 40),
            MyBlock((40, 8, 8), 40, 40),
            MyBlock((40, 8, 8), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 4, 4), 40, 20),
            MyBlock((20, 4, 4), 20, 20),
            MyBlock((20, 4, 4), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 8, 8), 80, 40),
            MyBlock((40, 8, 8), 40, 20),
            MyBlock((20, 8, 8), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 16, 16), 40, 20),
            MyBlock((20, 16, 16), 20, 10),
            MyBlock((10, 16, 16), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 32, 32), 20, 10),
            MyBlock((10, 32, 32), 10, 10),
            MyBlock((10, 32, 32), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 100, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

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


def training_loop(ddpm, loader, n_epochs, optim, device, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


ddpm = MyDDPM(MyUNet(), device=device)

total_params = len(torch.nn.utils.parameters_to_vector(ddpm.parameters()))
print(f'> Number of model parameters {total_params}')
if total_params > 1000000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

# Training
store_path = "ddpm_CIFAR100.pt"
if not params['no_train']:
    training_loop(ddpm, loader, params['n_epochs'], optim=Adam(ddpm.parameters(), params['lr']), device=device, store_path=store_path)


# Loading the trained model
best_model = MyDDPM(MyUNet(), n_steps=1000, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

# now show some interpolations (note you do not have to do linear interpolations as shown here, you can do non-linear or gradient-based interpolation if you wish)
#col_size = int(np.sqrt(params['batch_size']))
#
#z0 = z[0:col_size].repeat(col_size, 1)  # z for top row
#z1 = z[params['batch_size'] - col_size:].repeat(col_size, 1)  # z for bottom row
#
#t = torch.linspace(0, 1, col_size).unsqueeze(1).repeat(1, col_size).view(params['batch_size'], 1).to(device)
#
#lerp_z = (1 - t) * z0 + t * z1  # linearly interpolate between two points in the latent space
#lerp_g = Gen.sample(lerp_z)  # sample the model at the resulting interpolated latents
#
#plt.rcParams['figure.dpi'] = 100
#plt.grid(False)
#plt.imshow(torchvision.utils.make_grid(lerp_g).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
#plt.show()
#
## define directories
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
#num_generated = 0
#while num_generated < num_samples:
#
#    # sample from your model, you can modify this
#    z = torch.randn(params['batch_size'], params['n_latent']).to(device)
#    samples_batch = Gen.sample(z).cpu().detach()
#
#    for image in samples_batch:
#        if num_generated >= num_samples:
#            break
#        save_image(image, os.path.join(generated_images_dir, f"gen_img_{num_generated}.png"))
#        num_generated += 1
#
## save 10k images from the CIFAR-100 test dataset
#num_saved_real = 0
#while num_saved_real < num_samples:
#    real_samples_batch, _ = next(test_iterator)
#    for image in real_samples_batch:
#        if num_saved_real >= num_samples:
#            break
#        save_image(image, os.path.join(real_images_dir, f"real_img_{num_saved_real}.png"))
#        num_saved_real += 1
#
## compute FID
#score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
#print(f"FID score: {score}")
