'''
Author: Xiang Pan
Date: 2021-12-12 23:30:30
LastEditTime: 2021-12-14 01:49:16
LastEditors: Xiang Pan
Description: 
FilePath: /project/tools.py
@email: xiangpan@nyu.edu
'''
import torch
import numpy as np 
import torch.nn.functional as F

def mIOU(pred, label, num_classes=19):
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

# def pixel_accuracy(output, mask):
#     with torch.no_grad():
#         output = torch.argmax(F.softmax(output, dim=1), dim=1)
#         correct = torch.eq(output, mask).int()
#         accuracy = float(correct.sum()) / float(correct.numel())
#     return accuracy

# # def mIoU(pred_mask, mask, smooth=1e-10, n_classes=256):
# #     with torch.no_grad():
# #         # pred_mask = pred_mask.float()
# #         pred_mask = F.softmax(pred_mask, dim=1)
# #         pred_mask = torch.argmax(pred_mask, dim=1)
# #         pred_mask = pred_mask.contiguous().view(-1)
# #         mask = mask.contiguous().view(-1)

# #         iou_per_class = []
# #         for clas in range(0, n_classes): #loop per pixel class
# #             true_class = pred_mask == clas
# #             true_label = mask == clas

# #             if true_label.long().sum().item() == 0: #no exist label in this loop
# #                 iou_per_class.append(np.nan)
# #             else:
# #                 intersect = torch.logical_and(true_class, true_label).sum().float().item()
# #                 union = torch.logical_or(true_class, true_label).sum().float().item()

# #                 iou = (intersect + smooth) / (union +smooth)
# #                 iou_per_class.append(iou)
# #         return np.nanmean(iou_per_class)

# def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
#     pred[label == ignore_index] = 0
#     ious = []
#     for c in classes:
#         label_c = label == c
#         if only_present and np.sum(label_c.int().cpu().numpy()) == 0:
#             ious.append(np.nan)
#             continue
#         pred_c = pred == c
#         intersection = np.logical_and(pred_c.cpu(), label_c.cpu()).sum()
#         union = np.logical_or(pred_c.cpu(), label_c.cpu()).sum()
#         if union != 0:
#             ious.append(intersection / union)
#     mean = torch.Tensor(ious)
#     mean = mean[~mean.isnan()].mean()
#     return mean if ious else 0

# def iou_score(output, target):
#     smooth = 1e-6

#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()

#     return (intersection + smooth) / (union + smooth)


# def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    
    
# # Numpy version
# # Well, it's the same function, so I'm going to omit the comments

# def iou_numpy(outputs: np.array, labels: np.array):
#     outputs = outputs.squeeze(1)
    
#     intersection = (outputs & labels).sum((1, 2))
#     union = (outputs | labels).sum((1, 2))
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)
    
#     thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
#     return thresholded  # Or thresholded.mean()