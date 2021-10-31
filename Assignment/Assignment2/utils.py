'''
Author: Xiang Pan
Date: 2021-10-23 20:54:16
LastEditTime: 2021-10-28 23:19:50
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment2/utils.py
xiangpan@nyu.edu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from extras.shapes_loader import get_shapes_loader
from extras.util import *
import torch
import matplotlib.pyplot as plt
import matplotlib
from extras.encoder import ResnetEncoder
from extras.anchors import get_offsets
from extras.boxes import box_iou, nms

trainloader, valloader = get_shapes_loader(batch_sz=32)
device = 'cuda' # change this to "cuda" if you have access to a GPU. On CPU, training should take around ~45 mins on CPU.

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

def get_gt_boxes():
    """
    Generate 192 boxes where each box is represented by :
    [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Each anchor position should generate 3 boxes according to the scales and ratios given.

    Return this result as a numpy array of size [192,4]
    """
    stride = 16 # The stride of the final feature map is 16 (the model compresses the image from 128 x 128 to 8 x 8)
    map_sz = 128 # this is the length of height/width of the image

    scales = torch.tensor([40,50,60])
    ratios = torch.tensor([[1,1]]).view(1,2)
    
    gt_boxes = []
    # TODO
    for i in range(0, map_sz//stride):
        for j in range(0, map_sz//stride):
            for scale in scales:
                top_left_x = (i * stride + stride//2) - scale//2
                top_left_y = (j * stride + stride//2) - scale//2
                bottom_right_x = (i * stride + stride//2) + scale//2
                bottom_right_y = (j * stride + stride//2) + scale//2
                temp = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
                temp = torch.tensor([t for t in temp])
                temp = torch.tensor([max(0,t) for t in temp])
                gt_boxes.append(temp)
                # print(temp)
    gt_boxes = torch.stack(gt_boxes)
    return gt_boxes

def get_bbox_gt(ex_boxes, gt_boxes, sz=128):
    '''
    
    INPUT:
    ex_boxes: [Nx4]: Bounding boxes in the image. Here N is the number of bounding boxes the image has
    gt_boxes: [192 x 4]: Anchor boxes of an image of size 128 x 128 with stride 16. 
    sz : 128
    OUTPUT: 
    gt_classes: [192 x 1] : Class labels for each anchor: 1 is for foreground, 0 is for background and -1 is for a bad anchor. [where IOU is between 0.3 and 0.7]
    gt_offsets: [192 x 4]: Offsets for anchor to best fit the bounding box object. 0 values for 0 and -1 class anchors.
    '''
    ex_boxes = ex_boxes.cpu()
    gt_boxes = gt_boxes.cpu()
    N = gt_boxes.shape[0]
    M = ex_boxes.shape[0]
    # gt_classes = []
    gt_offsets = torch.zeros(N, 4)
    high_threshold = 0.7
    low_threshold = 0.3
    def iou_fun(iou_val):
        if iou_val > 0.7:
            return 1
        elif iou_val > 0.3:
            return -1
        else:
            return 0
    def label_fun(iou_vec):
        if any(iou_vec == 1):
            # print(iou_vec, any(iou_vec))
            return 1
        elif any(iou_vec== -1):
            return -1
        else:
            return 0

    # print(gt_boxes.shape, ex_boxes.shape)
    iou_res = box_iou(gt_boxes, ex_boxes)
    assert(iou_res.shape == (N, M))
    gt_classes = iou_res.apply_(iou_fun).long()

    for i,j in (gt_classes == 1).nonzero():
        t = get_offsets(gt_boxes[i].unsqueeze(dim = 0), ex_boxes[j].unsqueeze(dim = 0)).squeeze()
        gt_offsets[i] = t
    gt_classes = torch.Tensor([label_fun(i) for i in gt_classes]).unsqueeze(dim = 1)
    # print(gt_offsets)
    assert(gt_classes.shape == (N, 1))
    assert(gt_offsets.shape == (N, 4))
    return gt_classes, gt_offsets


gt_boxes = get_gt_boxes()
def get_targets(target, sample):
    '''
    Input
    target => Set of bounding boxes for each image.
    Sample => Each image
    Output:
    Bounding box offsets and class labels for each anchor.
    '''
    batched_preds = []
    batched_offsets = []
    final_cls_targets = []
    final_box_offsets = []
    for t, s in zip(target, sample):
        bboxes = t['bounding_box'].to(device).float()
        class_targets, box_offsets = get_bbox_gt(bboxes, gt_boxes, sz=128)
        final_cls_targets.append(class_targets)
        final_box_offsets.append(box_offsets)
    
    final_cls_targets = torch.stack(final_cls_targets, dim=0)
    final_box_offsets = torch.stack(final_box_offsets, dim=0)

    return final_cls_targets, final_box_offsets


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int, int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            BCE_loss = criterion(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def class_loss(out_pred, class_targets, loss_type="BCE", weight=None, sampling_mask=None):
    if loss_type == "focal_loss":
        class_targets = class_targets.cuda()
        focal_loss = FocalLoss(gamma=2, alpha=0.25)
        mask = torch.zeros_like(class_targets.cuda())
        mask[(class_targets >= 0).nonzero(as_tuple=True)] = 1
        criterion = focal_loss
        out_pred = out_pred.mul(mask)
        class_targets = class_targets.mul(mask)
        sigmoid = nn.Sigmoid()
        # logit = torch.stack([1 - sigmoid(out_pred), sigmoid(out_pred)], dim=2)
        # logit = logit.reshape(-1, 2).cuda()
        # class_targets = class_targets.reshape(-1)
        # loss = criterion(logit, class_targets.long())
        loss = criterion(out_pred, class_targets)
    elif loss_type == "BCE":
        class_targets = class_targets.cuda()
        mask = torch.zeros_like(class_targets.cuda())
        mask[(class_targets >= 0).nonzero(as_tuple=True)] = 1
        if weight is not None:
            criterion = nn.BCEWithLogitsLoss(weight=weight)
        else:
            if sampling_mask is not None:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
            else:
                criterion = nn.BCEWithLogitsLoss()
        out_pred = out_pred.mul(mask)
        class_targets = class_targets.mul(mask)
        sigmoid = nn.Sigmoid()
        # logit = torch.stack([1 - sigmoid(out_pred), sigmoid(out_pred)], dim=2)
        # logit = logit.reshape(-1, 2).cuda()
        # class_targets = class_targets.reshape(-1)
        # print((sigmoid(out_pred)>0.7)[class_targets.nonzero(as_tuple=True)],class_targets[class_targets.nonzero(as_tuple=True)])
        loss = criterion(out_pred, class_targets)
        # print(loss.shape)
        if sampling_mask is not None:
            loss = loss.mul(sampling_mask)
            loss = loss.mean()
    return loss

def bbox_loss(out_bbox, box_targets, class_targets):
    # print(class_targets.shape)
    mask = torch.zeros_like(class_targets).cuda()
    mask[(class_targets == 1).nonzero(as_tuple=True)] = 1
    criterion = nn.SmoothL1Loss(reduction='none')
    loss = criterion(out_bbox.cuda(), box_targets.cuda()).mean(dim = 2).mul(mask).mean()
    return loss

if __name__ == "__main__":
    gt_boxes = get_gt_boxes()

    assert gt_boxes.size() == (192,4)