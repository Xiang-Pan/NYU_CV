'''
Author: Xiang Pan
Date: 2021-09-26 21:43:41
LastEditTime: 2021-09-30 00:56:14
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/option.py
xiangpan@nyu.edu
'''
import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False

parser = argparse.ArgumentParser()
parser.add_argument('--name',           type=str,    default=None,       required=False,)
parser.add_argument('--learning_rate',  type=float,  default=0.01,       required=False,)
parser.add_argument('--batch_size',     type=int,    default=32,         required=False,)
parser.add_argument('--warmup_epochs',  type=int,    default=-1,         required=False,)
parser.add_argument('--max_epochs',     type=int,    default=80,         required=False,)
parser.add_argument('--backbone_name',  type=str,    default="ResNet18", required=False,)
parser.add_argument('--scheduler_name', type=str,    default=None,       required=False,)
parser.add_argument('--optimizer_name', type=str,    default="adam",     required=False,)
parser.add_argument('--weight_decay',   type=float,  default=1e-5,       required=False,)
parser.add_argument('--aug',            action='store_true',             default=False,)
parser.add_argument('--mixup',          action='store_true',             default=False,)
parser.add_argument('--label_smoothing',type=float,  default=False,)
parser.add_argument('--focal_loss',     action='store_true',             default=False,)

def get_option():
    option = parser.parse_args()
    return option