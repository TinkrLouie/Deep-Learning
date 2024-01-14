import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from IPython import display as disp
from cleanfid import fid
from torchvision.utils import save_image
# import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# let's view some of the training data
plt.rcParams['figure.dpi'] = 100
x, t = next(train_iterator)
x, t = x.to(device), t.to(device)
plt.imshow(torchvision.utils.make_grid(x).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
plt.show()


class Autoencoder(nn.Module):
    def __init__(self, params, f=32):
        super(Autoencoder, self).__init__()
        # self.encoder = nn.Linear(32 * 32 * params['n_channels'], params['n_latent'])
        # self.decoder = nn.Linear(params['n_latent'], 32 * 32 * params['n_channels'])

        self.encoder = nn.Sequential(
            nn.Conv2d(params['n_channels'], f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # output = 16x16
            nn.Conv2d(f, f * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # output = 8x8
            nn.Conv2d(f * 2, f * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # output = 4x4
            nn.Conv2d(f * 4, f * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # output = 2x2
            nn.Conv2d(f * 4, f * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # output = 1x1
            nn.Conv2d(f * 4, params['n_latent'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(params['n_latent']),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),  # output = 2x2
            nn.Conv2d(params['n_latent'], f * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # output = 4x4
            nn.Conv2d(f * 4, f * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # output = 8x8
            nn.Conv2d(f * 4, f * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # output = 16x16
            nn.Conv2d(f * 2, f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # output = 32x32
            nn.Conv2d(f, params['n_channels'], 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(x.size(0), -1))
        z += 0.1 * torch.randn_like(z)  # crude attempt to make latent space normally distributed so we can sample it
        x = torch.sigmoid(self.decoder(z))
        return x.view(x.size(0), params['n_channels'], 32, 32)

    def sample(self, z):  # sample from some prior distribution (it should not depend on x)
        x = torch.sigmoid(self.decoder(z))
        return x.view(x.size(0), params['n_channels'], 32, 32)


# hyperparameters
params = {
    'batch_size': train_loader.batch_size,
    'n_channels': 3,
    'n_latent': 512  # alters number of parameters
}

N = Autoencoder(params).to(device)

print(f'> Number of model parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')
if len(torch.nn.utils.parameters_to_vector(N.parameters())) > 1000000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

# Loss function
loss_fn = nn.MSELoss(reduction='sum')

# initialise the optimiser
optimiser = torch.optim.SGD(N.parameters(), lr=0.001, momentum=0.9)
steps = 0

# keep within our optimisation step budget
while steps < 50000:

    # arrays for metrics
    loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(1000):
        x, t = next(train_iterator)
        x, t = x.to(device), t.to(device)

        # train model
        p = N(x)
        loss = loss_fn(p, x)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        steps += 1

        loss_arr = np.append(loss_arr, loss.item())

    print('steps {:.2f}, loss: {:.3f}'.format(steps, loss_arr.mean()))

    # sample model and visualise results (ensure your sampling code does not use x)
    N.eval()
    z = torch.randn(params['batch_size'], params['n_latent']).to(device)
    samples = N.sample(z).cpu().detach()
    plt.imshow(torchvision.utils.make_grid(samples).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
    plt.show()
    disp.clear_output(wait=True)
    N.train()

# now show some interpolations (note you do not have to do linear interpolations as shown here, you can do non-linear or gradient-based interpolation if you wish)
col_size = int(np.sqrt(params['batch_size']))

z0 = z[0:col_size].repeat(col_size, 1)  # z for top row
z1 = z[params['batch_size'] - col_size:].repeat(col_size, 1)  # z for bottom row

t = torch.linspace(0, 1, col_size).unsqueeze(1).repeat(1, col_size).view(params['batch_size'], 1).to(device)

lerp_z = (1 - t) * z0 + t * z1  # linearly interpolate between two points in the latent space
lerp_g = N.sample(lerp_z)  # sample the model at the resulting interpolated latents

plt.rcParams['figure.dpi'] = 100
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(lerp_g).cpu().numpy().transpose(1, 2, 0), cmap=plt.cm.binary)
plt.show()


# define directories
real_images_dir = 'real_images'
generated_images_dir = 'generated_images'
num_samples = 10000  # do not change


# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


setup_directory(real_images_dir)
setup_directory(generated_images_dir)

# generate and save 10k model samples
num_generated = 0
while num_generated < num_samples:

    # sample from your model, you can modify this
    z = torch.randn(params['batch_size'], params['n_latent']).to(device)
    samples_batch = N.sample(z).cpu().detach()

    for image in samples_batch:
        if num_generated >= num_samples:
            break
        save_image(image, os.path.join(generated_images_dir, f"gen_img_{num_generated}.png"))
        num_generated += 1

# save 10k images from the CIFAR-100 test dataset
num_saved_real = 0
while num_saved_real < num_samples:
    real_samples_batch, _ = next(test_iterator)
    for image in real_samples_batch:
        if num_saved_real >= num_samples:
            break
        save_image(image, os.path.join(real_images_dir, f"real_img_{num_saved_real}.png"))
        num_saved_real += 1

# compute FID
score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
print(f"FID score: {score}")
