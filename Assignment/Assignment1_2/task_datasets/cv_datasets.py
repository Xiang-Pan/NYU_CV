'''
Author: Xiang Pan
Date: 2021-09-09 17:23:15
LastEditTime: 2021-09-09 17:51:37
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/task_datasets/cv_datasets.py
xiangpan@nyu.edu
'''
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# batch_size = 32
# momentum = 0.9
# lr = 0.01
# epochs = 10
# log_interval = 10

class CVDataset(Dataset):
    def __init__(self, split, transform = None):
        self.split = split
        if split in ["train","validation"]:
            path = "cached_datasets/"+split+"/"
            X_path = path + "X.pt"
            y_path = path + "y.pt"
            self.X = torch.load(X_path).squeeze(1)
            self.y = torch.load(y_path).squeeze(1)
        elif split == "testing":
            path = "cached_datasets/"+split+"/"
            X_path = path + "test.pt"
            # y_path = path + "y.pt"
            self.X = torch.load(X_path).squeeze(1)
            # self.y = torch.load(y_path).squeeze(1)
        else:
            raise ValueError("Invalid split")
        if transform:
            print("transform")
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        if self.split != "testing":
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

def get_cv_dataloader(batch_size = 32):
    train_dataset = CVDataset("train")
    test_dataset = CVDataset("testing")
    val_dataset = CVDataset("validation")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset = CVDataset("train")
    dataset[0]
    print(dataset[0][0].shape)