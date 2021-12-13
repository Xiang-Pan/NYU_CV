'''
Author: Xiang Pan
Date: 2021-10-23 20:56:23
LastEditTime: 2021-10-28 23:17:00
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment2/main.py
xiangpan@nyu.edu
'''
from extras.shapes_loader import get_shapes_loader
from extras.util import *
import torch
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from extras.encoder import ResnetEncoder
import torch.nn.functional as F
from extras.anchors import get_offsets
from extras.boxes import box_iou, nms
from torchmetrics import ConfusionMatrix
import random

from models import *
from utils import *

def train(ep, model, trainloader, optimizer, loss_type="BCE"):
    total_loss = 0
    b_loss = 0
    c_loss = 0
    confmat = ConfusionMatrix(num_classes=2)
    for i, (ims, targets) in enumerate(trainloader):
        ims = torch.stack(list(ims), dim=0).to(device)
        class_targets, box_targets = get_targets(targets, ims)
        class_targets = class_targets.squeeze()
        
        
        positive_index = (class_targets==1).nonzero(as_tuple=True)
        negative_index = (class_targets==0).nonzero()
        positive_number = (class_targets==1).nonzero().size()[0]
        negative_number = class_targets.shape[0] * class_targets.shape[1] - positive_number

        a = negative_index
        a = a[torch.randperm(a.size()[0])]
        kept_neg = a[:positive_number]
        masked_neg = a[positive_number:]
        sampling_mask = torch.ones_like(class_targets).to(device)
        masked_neg = tuple(masked_neg.T)
        # print(masked_neg)
        sampling_mask[masked_neg] = 0

        out_pred, out_box = model(ims)
        loss_cls = class_loss(out_pred, class_targets, loss_type=loss_type, sampling_mask=sampling_mask)
        loss_bbox = bbox_loss(out_box, box_targets, class_targets)
        # confmat(out_pred, target)
        # loss_bbox = 0
        lamda = 1
        loss = loss_cls + lamda * loss_bbox
        sigmoid = nn.Sigmoid()
        print((sigmoid(out_pred) > 0.7).nonzero())


        if loss.item() != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        c_loss += loss_cls.item()
        b_loss += loss_bbox.item()
        print(total_loss / (i+1), c_loss / (i+1), b_loss / (i+1))
        

    avg_c_loss = float(c_loss / len(trainloader))
    avg_b_loss = float(b_loss / len(trainloader))
    torch.save(model, "./cached_models/mask_lr_0.1_lamda1_"+str(ep))
    print('Trained Epoch: {} | Avg Classification Loss: {}, Bounding loss: {}'.format(ep, avg_c_loss, avg_b_loss))
# device = "cuda"
model = ShapesModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
start_epoch = 0
train_epoch = 20
loss_type = "BCE"
for ep in range(start_epoch, start_epoch + train_epoch):
    train(ep, model, trainloader, optimizer, loss_type)