'''
Author: Xiang Pan
Date: 2021-10-23 20:55:23
LastEditTime: 2021-10-23 20:55:23
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment2/model.py
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


class ShapesModel(nn.Module):

    def __init__(self):
        super(ShapesModel, self).__init__()

        # for each grid in the feature map we have 3 anchors of sizes: 40x40, 50x50, 60x60
        num_anchors = 3

        # regular resnet 18 encoder
        self.encoder = ResnetEncoder(num_layers=18, pretrained=False)

        # a small conv net
        self.conv = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1
        )

        # TODO: Add a Convolutional Layer to prediction the class predictions. This is a head that predicts whether a chunk/anchor contains an object or not.
        self.cls_logits =  nn.Conv2d(256, 3, kernel_size=1)

        # TODO: Add a Convolutional Layer to prediction the class predictions. This is a head that regresses over the 4 bounding box offsets for each anchor
        self.bbox_pred = nn.Conv2d(256, 12, kernel_size=1)
    
    def permute_and_flatten(self, layer, N, A, C, H, W):
        # helper function that rearranges the input for the loss function
        layer = layer.view(N, -1, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        layer = layer.reshape(N, -1, C)
        return layer
    
    def get_predict_regressions(self, cls_pred, box_pred):
        # helper function that gets outputs in the right shape for applying the loss
        N, AxC, H, W = cls_pred.shape
        Ax4 = box_pred.shape[1]
        A = Ax4 // 4
        C = AxC // A
        cls_pred = self.permute_and_flatten(
            cls_pred, N, A, C, H, W
        )
        
        box_pred = self.permute_and_flatten(
            box_pred, N, A, 4, H, W
        )
        return cls_pred, box_pred

    def forward(self, x):
        # x.size() == (1, 3, 128, 128)
        bt_sz = x.size(0)
        x = self.encoder(x)[3]
        x = F.relu(self.conv(x))
        # print(x.shape)
        # x.size() == (1, 256, 8, 8)
        cls_pred = self.cls_logits(x)
        # print(cls_pred.shape)
        # cls_pred.size() == (1, 3, 8, 8)
        box_pred = self.bbox_pred(x)
        # print(box_pred.shape)
        # box_pred.size() == (1, 12, 8, 8)
        cls_pred, box_pred = self.get_predict_regressions(cls_pred, box_pred)
        
        # cls_pred.size() == (1, 192, 1)
        # box_pred.size() == (1, 192, 4)
        return cls_pred.squeeze(2), box_pred

