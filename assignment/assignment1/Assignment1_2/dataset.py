'''
Author: Xiang Pan
Date: 2021-09-29 19:57:23
LastEditTime: 2021-09-29 20:22:47
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/dataset.py
xiangpan@nyu.edu
'''
from task_datasets.cv_datasets import *
from utils import *
from nets import *
from option import *
import pickle
import pandas as pd

args = get_option()
model = NET(backbone_name=args.backbone_name, num_classes=43, pretrained=False).cuda()
# train_dataset, val_dataset, test_dataset = get_cv_dataset(batch_size=32)

train_dataloader, val_dataloader, test_dataloader = get_cv_dataloader(batch_size = args.batch_size)
# dataframe_dict = {"Filename" : [], "ClassId": []}
# for batch_idx, () in range(batch_size):
    
file_ids = pickle.load(open('./cached_datasets/testing/file_ids.pkl', 'rb'))
# print(file_ids)
