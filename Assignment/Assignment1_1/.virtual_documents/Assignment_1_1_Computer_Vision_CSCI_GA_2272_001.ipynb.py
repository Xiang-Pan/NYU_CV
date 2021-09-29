from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# options
dataset = 'mnist' # options: 'mnist' | 'cifar10'
batch_size = 64   # input batch size for training
epochs = 10       # number of epochs to train
lr = 0.01        # learning rate


# Data Loading
# Warning: this cell might take some time when you run it for the first time, 
#          because it will download the datasets from the internet
dataset = "mnist"
def get_dataloader(dataset = "mnist"):
    if dataset == 'mnist':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root='.', train=True, download=True, transform=data_transform)
        testset = datasets.MNIST(root='.', train=False, download=True, transform=data_transform)

    elif dataset == 'cifar10':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=data_transform)
        testset = datasets.CIFAR10(root='.', train=False, download=True, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


import matplotlib.pyplot as plt

# image index is within trainset
image_index = 0
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = datasets.MNIST(root='.', train=True, download=True, transform=data_transform)

image, label = trainset[image_index]
image = image.reshape((28, 28))

# # Plot
plt.title('Label is {label}'.format(label=label))
plt.imshow(image, cmap='gray')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
# image index is within trainset
image_index = 1

trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=None)
image, label = trainset[image_index]

# # Plot
plt.title('Label is {label}'.format(label=label))
plt.imshow(image)


## network and optimizer
if dataset == 'mnist':
    num_inputs = 784
elif dataset == 'cifar10':
    num_inputs = 3072

num_outputs = 10 # same for both CIFAR10 and MNIST, both have 10 classes as outputs
#   - Convolution with 5 by 5 filters, 16 feature maps + Tanh nonlinearity.
#   - 2 by 2 max pooling (non-overlapping).
#   - Convolution with 5 by 5 filters, 128 feature maps + Tanh nonlinearity.
#   - 2 by 2 max pooling (non-overlapping).
#   - Flatten to vector.
#   - Linear layer with 64 hidden units + Tanh nonlinearity.
#   - Linear layer to 10 output units.
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, input):
        input = input.view(-1, num_inputs) # reshape input to batch x num_inputs
        output = self.linear(input)
        return output

network = Net(num_inputs, num_outputs)
optimizer = optim.SGD(network.parameters(), lr=lr)


def train(epoch, train_loader):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}get_ipython().run_line_magic(")]\tLoss:", " {:.6f}'.format(")
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        output = network(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        #test_loss += F.cross_entropy(output, target, sum=True).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}get_ipython().run_line_magic(")\n'.format(", "")
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))






rain_loader, test_loader 
for epoch in range(50):
    train(epoch)


import visdom
visdom.image(network.linear.weight)


for epoch in range(50):
    train(epoch)


test()


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLP, self).__init__()
        self.net = nn.Sequential( nn.Linear(num_inputs, 1000), nn.Linear(1000, num_outputs))

    def forward(self, input):
        input = input.view(-1, num_inputs) # reshape input to batch x num_inputs
        output = self.net(input)
        return output


network = MLP(num_inputs, num_outputs)
optimizer = optim.SGD(network.parameters(), lr=lr)


for epoch in range(50):
    train(epoch)


network = MLP(num_inputs, num_outputs)
optimizer = optim.SGD(network.parameters(), lr=10)


for epoch in range(50):
    train(epoch)


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CNN, self).__init__()
        self.net = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(1,-1),
            nn.Linear(75,64),
            nn.Tanh(),
            nn.Linear(64,10)
        )

    def forward(self, input):
        output = self.net(input)
        return output
    


network = CNN(num_inputs, num_outputs)
optimizer = optim.SGD(network.parameters(), lr=0.01)
train_loader, test_loader = get_dataloader("cifar10")
