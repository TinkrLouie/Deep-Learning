import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import *
import torchvision
from cleanfid import fid
import torchvision.utils as vutils
from torchvision.utils import save_image
# import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# hyperparameters
params = {
    'batch_size': train_loader.batch_size,
    'n_channels': 3,
    'n_latent': 32,
    'ngpu': 1,
    'lr': 0.0001,
    'gen_lr': 0.0001,
    'dis_lr': 0.0004,
    'n_epoch': 50000,
    'nz': 100,  # Size of z latent vector
    'real_label': 0.9,  # Label smoothing
    'fake_label': 0
}

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = Sequential(
            # input is Z, going into a convolution
            ConvTranspose2d(params['nz'], params['n_latent'] * 8, 3, 1, 0, bias=False),
            BatchNorm2d(params['n_latent'] * 8),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(params['n_latent'] * 8, params['n_latent'] * 4, 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent'] * 4),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(params['n_latent'] * 4, params['n_latent'] * 2, 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent'] * 2),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(params['n_latent'] * 2, params['n_latent'], 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent']),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(params['n_latent'], params['n_channels'], 3, 1, 1, bias=False),
            Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def sample(self, z):  # sample from some prior distribution (it should not depend on x)
        with torch.no_grad():
            return self.main(z)


# custom weights initialization called on netG and netD
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = Sequential(
            # input is (nc) x 64 x 64
            Conv2d(params['n_channels'], params['n_latent'], 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent']),
            LeakyReLU(0.2, inplace=True),
            Conv2d(params['n_latent'], params['n_latent'] * 2, 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent'] * 2),
            LeakyReLU(0.2, inplace=True),
            Conv2d(params['n_latent'] * 2, params['n_latent'] * 4, 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent'] * 4),
            LeakyReLU(0.2, inplace=True),
            Conv2d(params['n_latent'] * 4, params['n_latent'] * 8, 3, 1, 1, bias=False),
            BatchNorm2d(params['n_latent'] * 8),
            LeakyReLU(0.2, inplace=True),
            Conv2d(params['n_latent'] * 8, 1, 3, 1, 0, bias=False),
            AdaptiveAvgPool2d((1, 1)),
            Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

    def sample(self, z):
        with torch.no_grad():
            return self.main(z)


Gen = Generator().to(device)
Gen.apply(weights_init)
print(Gen)
Dis = Discriminator().to(device)
Dis.apply(weights_init)
print(Dis)

total_params = len(torch.nn.utils.parameters_to_vector(Gen.parameters())) + len(
    torch.nn.utils.parameters_to_vector(Dis.parameters()))
print(f'> Number of model parameters {total_params}')
if total_params > 1000000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

# Loss function
loss_fn = BCELoss()  # can change to reduction='sum'

# initialise the optimiser
Gen_optim = torch.optim.SGD(Gen.parameters(), lr=params['gen_lr'], momentum=0.9)
Dis_optim = torch.optim.SGD(Dis.parameters(), lr=params['dis_lr'], momentum=0.9)

steps = 0

loss_gen = []
loss_dis = []
img_list = []

# keep within our optimisation step budget
while steps < params['n_epoch']:

    # arrays for metrics

    # iterate over some of the train dateset
    for i in range(1000):
        x, t = next(train_iterator)
        x, t = x.to(device), t.to(device)

        b_size = x.size(0)
        # Apply label smoothing
        label = torch.full((b_size,), params['real_label'], device=device)

        # Add noise
        x = 0.9 * x + 0.1 * torch.randn((x.size()), device=device)

        # Train discriminator model
        Dis_optim.zero_grad()
        d = Dis(x).view(-1)  # Flatten
        #print(d.size(), label.size(), t.size())

        dis_loss_real = loss_fn(d, label)
        dis_loss_real.backward()

        # Generate with Generator model
        noise = torch.randn(params['batch_size'], params['nz'], 1, 1, device=device)
        gen_img = Gen(noise)
        # Fill label with 0, no smoothing. If smoothing, change param fake_label to 0.1
        label.fill_(params['fake_label'])

        gen_img = 0.9 * gen_img + 0.1 * torch.randn((gen_img.size()), device=device)
        d2 = Dis(gen_img.detach()).view(-1)  # Flatten
        dis_loss_fake = loss_fn(d2, label)
        dis_loss_fake.backward()
        dis_loss = dis_loss_fake + dis_loss_real
        # Update discriminator model
        Dis_optim.step()

        # Train Generator model
        Gen_optim.zero_grad()
        label.fill_(params['real_label'])
        d3 = Dis(gen_img).view(-1)  # Flatten
        gen_loss = loss_fn(d3, label)

        # Average confidence of Discriminator model
        conf = d3.mean().item()

        gen_loss.backward()
        Gen_optim.step()

        if steps % 500:
            Gen.eval()
            z = torch.randn(params['n_latent'], params['nz'], 1, 1).to(device)
            samples = Gen.sample(z).cpu().detach()
            img_list.append(vutils.make_grid(samples, padding=2, normalize=True))
        steps += 1

    loss_dis.append(dis_loss.item())
    loss_gen.append(gen_loss.item())
    print('steps {:.2f}, dis loss: {:.3f}, gen loss: {:.3f}'.format(steps, dis_loss.mean(), gen_loss.mean()))

# now show some interpolations (note you do not have to do linear interpolations as shown here, you can do non-linear or gradient-based interpolation if you wish)
col_size = int(np.sqrt(params['batch_size']))

z0 = z[0:col_size].repeat(col_size, 1)  # z for top row
z1 = z[params['batch_size'] - col_size:].repeat(col_size, 1)  # z for bottom row

t = torch.linspace(0, 1, col_size).unsqueeze(1).repeat(1, col_size).view(params['batch_size'], 1).to(device)

lerp_z = (1 - t) * z0 + t * z1  # linearly interpolate between two points in the latent space
lerp_g = Gen.sample(lerp_z)  # sample the model at the resulting interpolated latents

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
    samples_batch = Gen.sample(z).cpu().detach()

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
