'''
Author: Xiang Pan
Date: 2021-12-13 00:54:01
LastEditTime: 2021-12-13 00:55:02
LastEditors: Xiang Pan
Description: 
FilePath: /project/debug.py
@email: xiangpan@nyu.edu
'''
import torch
from torchmetrics import JaccardIndex
predictions = torch.load("./predictions.pt")
mask = torch.load("./mask.pt")
jaccard = JaccardIndex(num_classes=19)
jaccard(predictions, mask)