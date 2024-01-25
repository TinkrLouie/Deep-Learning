import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomRotation, Normalize
import os
from torchvision.datasets import CIFAR100
from torch.optim import SGD

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
lr = 0.01
valid_size = 0.2


def RGBshow(img):
    img = img * 0.5 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))


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


#train_length = len(train_dataset)
#indices = list(range(len(train_dataset)))
#split = int(np.floor(valid_size * train_length))
#
#np.random.shuffle(indices)
#
#train_idx = indices[split:]
#valid_idx = indices[:split]
#
#train_sampler = SubsetRandomSampler(train_idx)
#validation_sampler = SubsetRandomSampler(valid_idx)


# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


#train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
#valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)
#test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
train_iterator = iter(cycle(train_loader))
#valid_iterator = iter(cycle(valid_loader))
test_iterator = iter(cycle(test_loader))

print(f'Size of training dataset: {len(train_loader.dataset)}')
print(f'Size of testing dataset: {len(test_loader.dataset)}')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.b2 = nn.BatchNorm2d(48)
        self.b3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.b1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.b2(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.b3(self.conv5(x))))
        x = x.view(-1, 64)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.out(x)
        return x


def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = (1.0 / np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)


def train(model, lr):
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Number of epochs to train for
    loss_keeper = {'train': [], 'valid': []}
    acc_keeper = {'train': [], 'valid': []}
    train_class_correct = list(0. for _ in range(n_class))
    #valid_class_correct = list(0. for _ in range(n_class))
    class_total = list(0. for _ in range(n_class))

    # minimum validation loss ----- set initial minimum to infinity
    #valid_loss_min = np.Inf
    step = 0
    for epoch in range(n_epoch):
        train_loss = 0.0
        #valid_loss = 0.0

        model.train()
        #for images, labels in trainer:
        for _ in range(1000):
            images, labels = next(train_iterator)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            train_correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            for idx, label in enumerate(labels):
                train_class_correct[label] += train_correct[idx].item()
                class_total[label] += 1
            step += 1

        #model.eval()
        #for images, labels in validater:
        #    images, labels = images.to(device), labels.to(device)
        #    output = model(images)
        #    loss = criterion(output, labels)
        #    valid_loss += loss.item()
        #    _, pred = torch.max(output, 1)
        #    valid_correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        #    for idx, label in enumerate(labels):
        #        # for idx in range(batch_size):
        #        valid_class_correct[label] += valid_correct[idx].item()
        #        class_total[label] += 1

        # Calculating loss over entire batch size for every epoch
        train_loss = train_loss / len(train_loader)
        #valid_loss = valid_loss / len(validater)

        # Calculating loss over entire batch size for every epoch
        train_acc = float(100. * np.sum(train_class_correct) / np.sum(class_total))
        #valid_acc = float(100. * np.sum(valid_class_correct) / np.sum(class_total))

        # saving loss values
        loss_keeper['train'].append(train_loss)
        #loss_keeper['valid'].append(valid_loss)

        # saving acc values
        acc_keeper['train'].append(train_acc)
        #acc_keeper['valid'].append(valid_acc)

        print(f"Epoch : {epoch + 1}")
        #print(f"Training Loss : {train_loss}\tValidation Loss : {valid_loss}")
        print(f"Training Loss : {train_loss}")
        #if valid_loss <= valid_loss_min:
        #    print(f"Validation loss decreased from : {valid_loss_min} ----> {valid_loss} ----> Saving Model.......")
        #    z = type(model).__name__
        #    torch.save(model.state_dict(), z + '_model.pth')
        #    valid_loss_min = valid_loss

        #print(f"Training Accuracy : {train_acc}\tValidation Accuracy : {valid_acc}\n\n")
        print(f"Training Accuracy : {train_acc}\n\n")
    print(step)
    return loss_keeper, acc_keeper


def test(model):
    test_loss = 0
    class_correct = list(0. for _ in range(n_class))
    class_total = list(0. for _ in range(n_class))

    model.eval()  # test the model with dropout layers off
    #for images, labels in test_loader:
    for _ in range(1000):
        images, labels = next(test_iterator)
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

        for idx, label in enumerate(labels):
            class_correct[label] += correct[idx].item()
            class_total[label] += 1

    test_loss = test_loss / len(test_loader)
    print(f'For {type(model).__name__} :')
    print(f"Test Loss: {test_loss}")
    print(f"Correctly predicted per class : {class_correct}, Total correctly perdicted : {sum(class_correct)}")
    print(f"Total Predictions per class : {class_total}, Total predictions to be made : {sum(class_total)}\n")
    for i in range(100):
        if class_total[i] > 0:
            print(
                f"Test Accuracy of class {class_names[i]} : {float(100 * class_correct[i] / class_total[i])}% where {int(np.sum(class_correct[i]))} of {int(np.sum(class_total[i]))} were predicted correctly")
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_names[i]))

    print(
        f"\nOverall Test Accuracy : {float(100. * np.sum(class_correct) / np.sum(class_total))}% where {int(np.sum(class_correct))} of {int(np.sum(class_total))} were predicted correctly")


cnn = CNN().to(device)
# print the number of parameters - this should be included in your report
print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(cnn.parameters()))}')

if len(torch.nn.utils.parameters_to_vector(cnn.parameters())) > 100000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")

cnn.apply(weight_init_normal)

criterion = nn.CrossEntropyLoss()

loss, acc = train(cnn, lr)
test(cnn)

# TODO: data visualisation of train loss and accuracy
# TODO: reference existing code
# TODO: a plot of the training and test accuracy over the length of your training
# TODO: the total number of parameters in your network
# TODO: the final values for training loss, training accuracy and test accuracy (means and standard deviations)
