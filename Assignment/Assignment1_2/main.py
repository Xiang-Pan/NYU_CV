'''
Author: Xiang Pan
Date: 2021-09-09 17:21:28
LastEditTime: 2021-09-29 23:22:38
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/main.py
xiangpan@nyu.edu
'''
from nets import NET
import torch
import torch.nn as nn
import pickle
import pandas as pd
import torch.optim as optim
import os
import sys
import math
from cached_datasets.evaluate import *

from task_datasets.cv_datasets import get_cv_dataloader
import wandb
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from warmup_scheduler import GradualWarmupScheduler
from utils import *
from option import *
from sklearn.model_selection import KFold

wandb.init(project="assignment1_2")

torch.manual_seed(1)

def get_scheduler(optimizer, args):
    scheduler_name = args.scheduler_name
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs-args.warmup_epochs)
    elif scheduler_name == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[7])
    elif scheduler_name == 'ReduceLROnPlateau':
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5,factor=0.5,verbose=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
    else:
        scheduler = None
    if scheduler is not None:
        if args.warmup_epochs > 0:
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler)
            return scheduler
    return scheduler


def train(max_epochs, model, optimizer, train_loader, val_loader, test_loader, args):
    
    # print(len(train_loader))
    # print(len(train_loader))
    max_train_acc = 0
    max_val_acc = 0
    


    optimizer.zero_grad()
    optimizer.step()
    if args.label_smoothing > 0:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    scheduler = get_scheduler(optimizer, args)
    
    for cur_epoch in range(0, max_epochs):
        # scheduler_warmup.step(cur_epoch)
        # train
        model.train()
        train_acc_metric = torchmetrics.Accuracy().cuda()
        train_acc_metric.reset()
        train_loss_epoch = 0
        val_loss_epoch = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            train_loss = criterion(output, target)
            train_loss_epoch += train_loss.item()
            
            preds = output.softmax(dim=-1)
            acc = train_acc_metric(preds, target).cuda()
            
            wandb.log({'train_loss': train_loss})
            wandb.log({'train_acc': acc})

            if args.mixup:
                if cur_epoch > args.warmup_epochs:
                    mixed_x, y_a, y_b, lam = mixup_data(data, target)
                    mixed_pred = model(mixed_x)
                    mixup_loss = mixup_criterion(criterion, mixed_pred, y_a, y_b, lam)
                    train_loss += mixup_loss
                    wandb.log({'mixup_train_loss': mixup_loss.item()})
                
            
            train_loss.backward()
            optimizer.step()
            
        train_acc = train_acc_metric.compute()
        wandb.log({'train_acc_epoch': train_acc, "epoch": cur_epoch})
        wandb.log({'train_loss_epoch': train_loss_epoch/len(train_loader), "epoch": cur_epoch})
        max_train_acc = max(max_train_acc, train_acc)
        wandb.log({'max_train_acc': max_train_acc, "epoch": cur_epoch})

        # val
        model.eval()
        val_acc_metric = torchmetrics.Accuracy().cuda()
        val_acc_metric.reset()
        for batch_idx, (data, target) in enumerate(val_loader):
            
            data = data.cuda()
            target = target.cuda()
            output = model(data)

            val_loss = F.cross_entropy(output, target)
            val_loss_epoch += val_loss.item()
            preds = output.softmax(dim=-1)
            acc = val_acc_metric(preds, target).cuda()
            wandb.log({'val_loss': val_loss})
            wandb.log({'val_acc': acc})
        val_acc = val_acc_metric.compute()
        wandb.log({'val_acc_epoch': val_acc, "epoch": cur_epoch})
        wandb.log({'val_loss_epoch': val_loss_epoch/len(val_loader), "epoch": cur_epoch})
        max_val_acc = max(max_val_acc, val_acc)
        wandb.log({'mac_val_acc': max_val_acc, "epoch": cur_epoch})
        
        
        # predict test data
        path = "./outputs/"+wandb.run.id
        if not os.path.exists(path):
            os.system("mkdir -p %s" % path)
        outfile_name = path+"/"+str(cur_epoch)+".csv"
        model_name = path+"/"+str(cur_epoch)+".pt"

        output_file = open(outfile_name, "w")
        
        # dataframe_dict = {"Filename" : [], "ClassId": []}
        df = pd.DataFrame(columns=['Filename', 'ClassId'])
        file_ids = pickle.load(open('./cached_datasets/testing/file_ids.pkl', 'rb'))
        
        for batch_idx, data in enumerate(test_loader):
    
            data = data.cuda()
            output = model(data)
            preds = torch.argmax(output.softmax(dim=-1), dim = -1).cpu().detach().tolist()
            
            file_id = file_ids[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            dft = pd.DataFrame(columns=['Filename', 'ClassId'], data=list(zip(file_id, preds)))
            df = df.append(dft, ignore_index=True)

        test_acc = evaluate_pred_df(df)
        wandb.log({'test_acc': test_acc, "epoch": cur_epoch})
        df.to_csv(outfile_name, index=False)
        print("Written to csv file {}".format(outfile_name))
        torch.save(model, model_name)
        
        
        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                # scheduler.step(val_acc)
                scheduler.step(np.around(val_acc.cpu().numpy(), decimals=2))
                wandb.log({'learning_rate': optimizer.param_groups[0]['lr'], "epoch": cur_epoch})
            else:
                scheduler.step(cur_epoch)
                wandb.log({'learning_rate': scheduler.get_last_lr(), "epoch": cur_epoch})
                
def get_auto_name(args):
    if args.scheduler_name is None:
        scheduler_name = 'None'
    else:
        scheduler_name = args.scheduler_name
    auto_name = '_'.join([  args.backbone_name,
                            scheduler_name,
                            str(args.learning_rate),
                            str(args.max_epochs),
                            str(args.batch_size),
                            str(args.warmup_epochs),
                            str(args.weight_decay),
                            str(args.label_smoothing),
                        ])
    return auto_name

def main():
    # print(1)
    
    args = get_option()
    if args.name is None:
        wandb.run.name = get_auto_name(args)
    else:
        wandb.run.name = args.name
    
    nclasses = 43 # GTSRB has 43 classes
    model = NET(backbone_name=args.backbone_name, num_classes=43, pretrained=False)
    model.apply(weight_init)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(model.parameters(), betas=[0.9, 0.999], lr=args.learning_rate, eps=1e-8)
    # no_decay = list()
    # decay = list()
    # for m in model.modules():
    #     if isinstance(m, (nn.Linear, nn.Conv2d)):
    #         decay.append(m.weight)
    #         no_decay.append(m.bias)
    #     elif hasattr(m, 'weight'):
    #         no_decay.append(m.weight)
    #     elif hasattr(m, 'bias'):
    #         no_decay.append(m.bias)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)

    # {'params': no_decay, 'weight_decay', 0}, {'params': decay}, **kwargs]
    train_dataloader, val_dataloader, test_dataloader = get_cv_dataloader(batch_size = args.batch_size)
    train(args.max_epochs, model, optimizer, train_dataloader, val_dataloader, test_dataloader, args)


if __name__ == '__main__':
    main()