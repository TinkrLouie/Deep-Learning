import shutil
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_symbolic import Input, SymbolicModel
from pytorch_symbolic import useful_layers
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomRotation, Normalize, RandomCrop
import os
from torchvision.datasets import CIFAR100
from torch.optim import SGD

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))
my_path = os.path.abspath(__file__)

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

n_steps = 10000
batch_size = 64
n_channels = 3
dim = 32
n_class = 100
n_epoch = 10
lr = 0.01
weight_decay = 1e-4

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

transform_train = Compose([
    RandomCrop(32, padding=4, padding_mode='reflect'),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
])

# Normalize the test set same as training set without augmentation
transform_test = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

ds = CIFAR100
train_dataset = ds("./datasets", download=True, train=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = ds("./datasets", download=True, train=False, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)


# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


train_iterator = iter(cycle(train_loader))

print(f'Size of training dataset: {len(train_loader.dataset)}')
print(f'Size of testing dataset: {len(test_loader.dataset)}')


# TODO: Experiment kaiming normalisation
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = (1.0 / np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)
        #init.kaiming_normal_(m.weight)
    elif classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Model referenced from GitHub repo of Pytorch-Symbolic
# Reference: https://github.com/sjmikler/pytorch-symbolic/tree/main
def classifier(flow, n_classes, pooling="avgpool"):
    if pooling == "catpool":
        maxp = flow(nn.MaxPool2d(kernel_size=(flow.H, flow.W)))
        avgp = flow(nn.AvgPool2d(kernel_size=(flow.H, flow.W)))
        flow = maxp(useful_layers.ConcatLayer(dim=1), avgp)(nn.Flatten())
    if pooling == "avgpool":
        flow = flow(nn.AvgPool2d(kernel_size=(flow.H, flow.W)))(nn.Flatten())
    if pooling == "maxpool":
        flow = flow(nn.MaxPool2d(kernel_size=(flow.H, flow.W)))(nn.Flatten())
    return flow(nn.Linear(flow.features, n_classes))


def shortcut_func(x, channels, stride):
    if x.channels != channels or stride != 1:
        return x(nn.Conv2d(x.channels, channels, kernel_size=1, bias=False, stride=stride))
    else:
        return x


def ResNet(
    input_shape,
    n_classes,
    strides=(1, 2, 2),
    group_sizes=(2, 2, 2),
    channels=(16, 32, 40),
    activation=nn.ReLU(),
    final_pooling="avgpool",
    dropout=0,  # p for dropout layers
    bn_ends_block=False
):

    def simple_block(flow, channels, stride):
        if preactivate_block:
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        flow = flow(nn.Conv2d(flow.channels, channels, 3, stride, 1))
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)

        if dropout:
            flow = flow(nn.Dropout(p=dropout))
        flow = flow(nn.Conv2d(flow.channels, channels, 3, 1, 1))

        if bn_ends_block:
            flow = flow(nn.BatchNorm2d(flow.channels))(activation)
        return flow

    block = simple_block

    inputs = Input(batch_shape=input_shape)

    # Head of the network
    flow = inputs(nn.Conv2d(inputs.channels, 16, 3, 1, 1))

    # The residual block
    for group_size, width, stride in zip(group_sizes, channels, strides):
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)
        preactivate_block = False

        for _ in range(group_size):
            residual = block(flow, width, stride)
            shortcut = shortcut_func(flow, width, stride)
            flow = residual + shortcut
            preactivate_block = True
            stride = 1

    # The classifier
    flow = flow(nn.BatchNorm2d(flow.channels))(activation)
    outs = classifier(flow, n_classes, pooling=final_pooling)
    #outs = nn.LogSoftmax(dim=1)(outs)
    model = SymbolicModel(inputs=inputs, outputs=outs)
    return model


# Function to find lr in optim params for clipping during training
def get_lr(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']


# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # remove any existing (old) data
    os.makedirs(directory)


# Set up dir for graphs
training_res_dir = 'training_res_images'
setup_directory(training_res_dir)

cnn = ResNet([batch_size, n_channels, dim, dim], n_class, dropout=0.2).to(device)  # Add final_pooling='catpool' to change pooling mode


# print the number of parameters - this should be included in your report
print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(cnn.parameters()))}')

if len(torch.nn.utils.parameters_to_vector(cnn.parameters())) > 100000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")
cnn.apply(weight_init)
optimiser = SGD(cnn.parameters(), lr=lr, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, lr, epochs=n_epoch, steps_per_epoch=1000)
criterion = nn.CrossEntropyLoss()

plot_data = []
step = 0
while step < n_steps:

    # arrays for metrics
    train_loss_arr = np.zeros(0)
    train_acc_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    # iterate through some of the train dateset
    for _ in range(1000):
        # Get input from batch loader
        images, labels = next(train_iterator)
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        # Forward pass
        output = cnn(images)
        # Loss
        loss = criterion(output, labels)
        # Gradients
        loss.backward()
        # TODO: Explore grad clipping
        # Grad clipping
        nn.utils.clip_grad_value_(cnn.parameters(), 1)  # Alternative: 0.1
        # Update optimiser
        optimiser.step()
        step += 1
        # Update scheduler
        # lr_keeper.append(get_lr(optimiser))
        # scheduler.step()
        _, pred = torch.max(output, 1)

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(labels.view_as(pred)).float().mean().item())

    # iterate over the entire test dataset
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = cnn(images)
        loss = criterion(output, labels)
        _, pred = torch.max(output, 1)
        test_acc_arr = np.append(test_acc_arr, pred.data.eq(labels.view_as(pred)).float().mean().item())

    # print your loss and accuracy data - include this in the final report
    print('steps: {:.2f}, train loss: {:.3f}, train acc: {:.3f}±{:.3f}, test acc: {:.3f}±{:.3f}'.format(
        step, train_loss_arr.mean(), train_acc_arr.mean(), train_acc_arr.std(), test_acc_arr.mean(), test_acc_arr.std()))

    # plot your accuracy graph - add a graph like this in your final report
    plot_data.append([step, np.array(train_acc_arr).mean(), np.array(train_acc_arr).std(), np.array(test_acc_arr).mean(), np.array(test_acc_arr).std()])

plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:grey', label="Train accuracy")
plt.fill_between([x[0] for x in plot_data], [x[1]-x[2] for x in plot_data], [x[1]+x[2] for x in plot_data], alpha=0.2, color='tab:grey')
plt.plot([x[0] for x in plot_data], [x[3] for x in plot_data], '-', color='tab:purple', label="Test accuracy")
plt.fill_between([x[0] for x in plot_data], [x[3]-x[4] for x in plot_data], [x[3]+x[4] for x in plot_data], alpha=0.2, color='tab:purple')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.savefig('classifier_training_result.png')


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.imshow()
    plt.savefig('lrs_history.png')


