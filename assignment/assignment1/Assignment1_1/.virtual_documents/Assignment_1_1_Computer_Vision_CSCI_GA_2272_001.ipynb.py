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
    
    elif dataset == 'mnist_limited':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        torch.utils.data.Subset
        trainset = torch.utils.data.Subset(datasets.MNIST(root='.', train=True, download=True, transform=data_transform), indices = [i for i in range(50)])
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



plt.figure(figsize=(20, 20))
f, axarr = plt.subplots(1,10) 
plt.subplot_tool()

for i in range(10):
    image, label = trainset[i]
    image = image.reshape((28, 28))
    axarr[i].imshow(image, cmap='gray')

plt.show()


import matplotlib.pyplot as plt
import numpy as np
# image index is within trainset
image_index = 1

trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=None)
image, label = trainset[image_index]

# # # Plot
# plt.title()
# plt.imshow(image)

plt.figure(figsize=(20, 20))
f, axarr = plt.subplots(1,10) 
plt.subplot_tool()

for i in range(10):
    image, label = trainset[i]
    axarr[i].imshow(image, cmap='gray')

plt.show()


## network and optimizer
# if dataset == 'mnist':
#     num_inputs = 784
# elif dataset == 'cifar10':
#     num_inputs = 3072

num_outputs = 10 
class SingleLayerNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SingleLayerNet, self).__init__()
        self.num_inputs = num_inputs
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, input):
        input = input.view(-1, self.num_inputs) # reshape input to batch x num_inputs
        output = self.linear(input)
        return output

network = SingleLayerNet(num_inputs=784, num_outputs=10)
optimizer = optim.SGD(network.parameters(), lr=lr)


def train(cur_epoch, network, train_loader):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        device = next(network.parameters()).device
        # print(network)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}get_ipython().run_line_magic(")]\tLoss:", " {:.6f}'.format(")
                cur_epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        device = next(network.parameters()).device
        # print(network)
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        #test_loss += F.cross_entropy(output, target, sum=True).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}get_ipython().run_line_magic(")\n'.format(", "")
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



train_loader, test_loader = get_dataloader("mnist")
for epoch in range(10):
    train(epoch, network, train_loader)
    test(network, test_loader)


print(network)


# import visdom
print(network.linear.weight.shape)
viz_data = network.linear.weight.reshape(10, 28, 28).detach().numpy()


import torchvision
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


plt.figure()
f, axarr = plt.subplots(1,10) 
plt.subplot_tool()

for i in range(10):
    axarr[i].imshow(viz_data[i], cmap='gray')


# show(viz_data)
plt.figure()
f, axarr = plt.subplots(1,10) 

for i in range(10):
    axarr[i].imshow(viz_data[i])



train_loader, test_loader = get_dataloader("mnist_limited")
for epoch in range(10):
    train(epoch, network, train_loader)
    test(network, test_loader)


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLP, self).__init__()
        self.num_inputs = num_inputs
        self.net = nn.Sequential(nn.Linear(num_inputs, 1000), 
                                 nn.Tanh(),
                                 nn.Linear(1000, num_outputs))

    def forward(self, input):
        input = input.view(-1, self.num_inputs) # reshape input to batch x num_inputs
        output = self.net(input)
        return output


optimizer = optim.SGD(network.parameters(), lr=lr)


train_loader, test_loader = get_dataloader("mnist")
network = MLP(num_inputs=784, num_outputs=10)
optimizer = optim.SGD(network.parameters(), lr=lr)
for epoch in range(10):
    train(epoch, network, train_loader)
    test(network, test_loader)


train_loader, test_loader = get_dataloader("mnist")
network = MLP(num_inputs=784, num_outputs=10)
optimizer = optim.SGD(network.parameters(), lr=10)
for epoch in range(10):
    train(epoch, network, train_loader)
    test(network, test_loader)


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CNN, self).__init__()
        self.net = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(5,5)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(1,-1),
            nn.Linear(3200, 64),
            nn.Tanh(),
            nn.Linear(64,10)
        )

    def forward(self, input):
        output = self.net(input)
        # print(output.shape)
        return output
    


train_loader, test_loader = get_dataloader("cifar10")
network = CNN(num_inputs=3072, num_outputs=10).cuda()
optimizer = optim.SGD(network.parameters(), lr=0.01)
for epoch in range(20):
    train(epoch, network, train_loader)
    test(network, test_loader)


viz_data = model.net[0].weight
print(viz_data.shape)

plt.figure(figsize=(20, 20))
for i, filter in enumerate(viz_data):
    plt.subplot(4, 4, i+1) 
    plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
    plt.axis('off')
plt.show()


import torchvision.models as models
from torchsummary import summary

model = CNN(num_inputs=3072, num_outputs=10).cuda()
 
summary(model, (3, 32, 32))
# -1 represent the batch size here
