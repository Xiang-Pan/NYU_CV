'''
Author: Xiang Pan
Date: 2021-09-29 21:07:42
LastEditTime: 2021-09-30 04:16:48
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/task_datasets/evaluate.py
xiangpan@nyu.edu
'''
from torchmetrics import ConfusionMatrix
import pandas as pd
import torch
import wandb
df = pd.read_csv('./cached_datasets/Test.csv')
d = dict()
for index,row in df.iterrows():
    key = str(index).zfill(5)
    val = row['ClassId']
    d[key] = val


def evaluate_pred_df(pred_df):
    # pred_df = pd.read_csv(path)
    correct = 0
    wrong = 0
    confmat = ConfusionMatrix(num_classes=43)
    for index,row in pred_df.iterrows():
        pred = row['ClassId']
        label = d[str(row['Filename']).zfill(5)]
        confmat(torch.tensor([pred]), torch.tensor([label]))
        if pred == label:
            correct += 1
        else:
            wrong += 1
        # d[key] = val
    wandb.log({'confmat': confmat})
    return correct/(correct+wrong)
