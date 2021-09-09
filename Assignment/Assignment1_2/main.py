'''
Author: Xiang Pan
Date: 2021-09-09 17:21:28
LastEditTime: 2021-09-09 17:57:25
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/main.py
xiangpan@nyu.edu
'''
from net import NET
import torch.optim as optim

from task_datasets.cv_datasets import get_cv_dataloader
import wandb
import torch.nn.functional as F
import torchmetrics

wandb.init(project="assignment1_2")


def train(epoch, model, optimizer, train_loader, val_loader):
    for cur_epoch in range(0, epoch):
        # train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            train_loss = F.nll_loss(output, target)
            train_loss.backward()
            optimizer.step()
            wandb.log({'train_loss': train_loss, "epoch": cur_epoch})
        # val
        model.eval()
        acc_metric = torchmetrics.Accuracy()
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            val_loss = F.nll_loss(output, target)
            preds = output.softmax(dim=-1)
            acc = acc_metric(preds, target)
            # optimizer.step()
            wandb.log({'val_loss': val_loss, "epoch": cur_epoch})
            wandb.log({'val_acc': acc, "epoch": cur_epoch})
            
            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
        # train(epoch)
        # validation()
        # model_file = 'model_' + str(epoch) + '.pth'
        # torch.save(model.state_dict(), model_file)
        # print('\nSaved model to ' + model_file + '.')

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         output = model(data)
#         validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))

def main():
    # print(1)
    model = NET()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_dataloader, val_dataloader, test_dataloader = get_cv_dataloader()
    train(10, model, optimizer, train_dataloader, val_dataloader)
    


if __name__ == '__main__':
    main()