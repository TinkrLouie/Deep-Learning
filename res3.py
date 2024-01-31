import numpy as np
import torch
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

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = 64
n_channels = 3
dim = 32
n_class = 100
n_epoch = 10
lr = 0.1

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
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Normalize the test set same as training set without augmentation
transform_test = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    dropout=0,
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

    # BUILDING HEAD OF THE NETWORK
    flow = inputs(nn.Conv2d(inputs.channels, 16, 3, 1, 1))

    # BUILD THE RESIDUAL BLOCKS
    for group_size, width, stride in zip(group_sizes, channels, strides):
        flow = flow(nn.BatchNorm2d(flow.channels))(activation)
        preactivate_block = False

        for _ in range(group_size):
            residual = block(flow, width, stride)
            shortcut = shortcut_func(flow, width, stride)
            flow = residual + shortcut
            preactivate_block = True
            stride = 1

    # BUILDING THE CLASSIFIER
    flow = flow(nn.BatchNorm2d(flow.channels))(activation)
    outs = classifier(flow, n_classes, pooling=final_pooling)
    model = SymbolicModel(inputs=inputs, outputs=outs)
    return model


def get_lr(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']


# Reference: https://github.com/NvsYashwanth/CIFAR-10-Image-Classification/tree/master
def train(model):
    lr_keeper = []
    loss_keeper = []
    acc_keeper = []

    print('Training...\n')

    for epoch in range(n_epoch):
        train_loss = 0.0
        train_class_correct = list(0. for _ in range(n_class))
        class_total = list(0. for _ in range(n_class))
        per_class_acc = []
        model.train()
        for _ in range(1000):
            # Get input from batch loader
            images, labels = next(train_iterator)
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            # Forward pass
            output = model(images)
            # Loss
            loss = criterion(output, labels)
            # Gradients
            loss.backward()

            # TODO: Explore grad clipping
            # Grad clipping
            # nn.utils.clip_grad_value_(model.parameters(), 0.1)
            # Update optimiser
            optimiser.step()

            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            train_correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            for idx, label in enumerate(labels):
                train_class_correct[label] += train_correct[idx].item()
                class_total[label] += 1

        # Calculating loss over entire batch size for every epoch
        train_loss = train_loss / len(train_loader)

        # Calculating loss over entire batch size for every epoch
        train_acc = float(100. * np.sum(train_class_correct) / np.sum(class_total))

        # saving loss values
        loss_keeper.append(train_loss)

        # saving acc values
        acc_keeper.append(train_acc)

        for i in range(n_class):
            per_class_acc.append(train_class_correct[i] / class_total[i])

        # TODO: Explore scheduler
        # Update scheduler
        lr_keeper.append(get_lr(optimiser))
        scheduler.step()

        print(f"Epoch : {epoch + 1}")
        print(f"Training Loss : {train_loss}")
        #print(train_class_correct)
        #print(class_total)
        #print(len(per_class_acc), per_class_acc)
        print(f"Training Accuracy : {train_acc}") #, stdev : {np.std(per_class_acc)}\n")
        test(cnn)

    return loss_keeper, acc_keeper, lr_keeper


def test(model):
    test_loss = 0
    class_correct = list(0. for _ in range(n_class))
    class_total = list(0. for _ in range(n_class))
    per_class_acc = []
    model.eval()
    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

        for idx, label in enumerate(labels):
            class_correct[label] += correct[idx].item()
            class_total[label] += 1

    for i in range(n_class):
        per_class_acc.append(float(100. * class_correct[i] / class_total[i]))
    #print(class_correct)
    #print(class_total)
    test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy : {float(100. * np.sum(class_correct) / np.sum(class_total))}")  #, stdev : {np.std(per_class_acc)}\n\n")


cnn = ResNet([batch_size, n_channels, dim, dim], n_class).to(device)

# print the number of parameters - this should be included in your report
print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(cnn.parameters()))}')

if len(torch.nn.utils.parameters_to_vector(cnn.parameters())) > 100000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")
cnn.apply(weight_init)
optimiser = SGD(cnn.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[5, 7], gamma=0.03, last_epoch=-1)
criterion = nn.CrossEntropyLoss()

loss, acc, lrs = train(cnn)