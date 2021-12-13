'''
Author: Xiang Pan
Date: 2021-10-23 22:02:31
LastEditTime: 2021-10-28 20:38:20
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment2/test.py
xiangpan@nyu.edu
'''
import torch
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
from utils import *


trainloader, valloader = get_shapes_loader(batch_sz=32)
sample, target = iter(valloader).next()
sample = torch.stack(sample,dim=0)
sample = sample.cuda()

trainloader, valloader = get_shapes_loader(batch_sz=32)
model = torch.load("./cached_models/lr_0.01_lamda1_13")
out_pred, out_box = model(sample)
sigmoid = nn.Sigmoid()
index = (sigmoid(out_pred)>0.7).nonzero(as_tuple=True)
selected_offsets = out_box[index]
# print(index, selected_offsets)
def visPred(model, sample):
    #TODO: visualize your model predictions on the sample image.
    out_pred, out_box = model(sample)
    sigmoid = nn.Sigmoid()
    index = (sigmoid(out_pred)>0.7).nonzero()
    # print(index)
    selected_boxes_index = (sigmoid(out_pred)>0.7).nonzero(as_tuple=True)
    selected_offsets = out_box[selected_boxes_index]
    print(selected_offsets.shape)
    
    
    batched_boxes = [[] for _ in range(32)]
    batched_scores = [[] for _ in range(32)]
    gt_boxes = get_gt_boxes()
    gt_boxes = gt_boxes.cuda()
    batch_size = 32
    count = 0
    for i,j in index:
        offset = selected_offsets[count]
        # print(offset)
        batched_boxes[i].append(gt_boxes[j] + offset)
        batched_scores[i].append(sigmoid(out_pred)[i][j])
        count = count + 1
    # print((sigmoid(out_pred)>0.5).nonzero())
    for i in range(batch_size):
        if len(batched_boxes[i]) == 0:
            continue
        boxes = torch.stack(batched_boxes[i])
        scores =  torch.stack(batched_scores[i])
        kept_index = nms(boxes, scores, iou_threshold=0.3)
        batched_boxes[i] = boxes[kept_index]
        batched_scores[i] = scores[kept_index]
    # 
    
    sample = sample.cpu()
    show_sample = sample.permute(0,2,3,1)
    fig, axs = plt.subplots(2,2)

    axs[0,0].imshow(show_sample[0])
    for t in batched_boxes[0]:
        draw_box(axs[0,0], t.detach().cpu())

    axs[0,1].imshow(show_sample[1])
    for t in batched_boxes[1]:
        draw_box(axs[0,1], t.detach().cpu())

    axs[1,0].imshow(show_sample[2])
    for t in batched_boxes[2]:
        draw_box(axs[1,0], t.detach().cpu())

    axs[1,1].imshow(show_sample[3])
    for t in batched_boxes[3]:
        draw_box(axs[1,1], t.detach().cpu())
# model = torch.load("../cached_models/lr_0.01_lamda1_13")
# model = torch.load("../retrain/retrain5")
# model = torch.load("../cached_models/cached_models_lr0.1/3")
visPred(model, sample)
