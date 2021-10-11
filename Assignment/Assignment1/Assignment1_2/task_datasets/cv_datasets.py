'''
Author: Xiang Pan
Date: 2021-09-09 17:23:15
LastEditTime: 2021-09-30 17:32:37
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/task_datasets/cv_datasets.py
xiangpan@nyu.edu
'''
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils import *
from randaugment import RandAugment

# batch_size = 32
# momentum = 0.9
# lr = 0.01
# epochs = 10
# log_interval = 10

class CVDataset(Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        if split in ["train","validation"]:
            path = "cached_datasets/"+split+"/"
            X_path = path + "X.pt"
            y_path = path + "y.pt"
            self.X = torch.load(X_path).squeeze(1)
            self.y = torch.load(y_path).squeeze(1)
            print(self.X.shape)
        elif split == "testing":
            path = "cached_datasets/"+split+"/"
            X_path = path + "test.pt"
            # y_path = path + "y.pt"
            self.X = torch.load(X_path).squeeze(1)
            # self.y = torch.load(y_path).squeeze(1)
        else:
            raise ValueError("Invalid split")
        self.transform = transform
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        if self.split != "testing":
            if self.transform:
                return self.transform(self.X[idx]), self.y[idx]
            else:
                return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

def get_cv_dataset(batch_size = 32):
    train_dataset = CVDataset("train")
    test_dataset = CVDataset("testing")
    val_dataset = CVDataset("validation")
    return train_dataset, val_dataset, test_dataset

def get_cv_dataloader(batch_size = 32, augument = False):
    if augument:
        train_dataset = torch.utils.data.ConcatDataset\
                    (
                        [
                            CVDataset("train", transform=None),
                            # CVDataset("train", transform=data_random_aug),
                            CVDataset("train", transform=transforms.Compose([transforms.ColorJitter(brightness=5),])),
                            CVDataset("train", transform=transforms.Compose([transforms.ColorJitter(saturation=5),])),
                            CVDataset("train", transform=transforms.Compose([transforms.ColorJitter(contrast=5),])),
                            CVDataset("train", transform=transforms.Compose([transforms.ColorJitter(hue=0.4),])),
                            CVDataset("train", transform=transforms.Compose([transforms.RandomRotation(15),])),
                            CVDataset("train", transform=transforms.Compose([transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5),])),
                            CVDataset("train", transform=transforms.Compose([transforms.Compose([transforms.Grayscale(num_output_channels=3),])])),
                            CVDataset("train", transform=transforms.Compose([transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),])),
                            CVDataset("train", transform=transforms.Compose([transforms.RandomAffine(degrees=15,shear=2),])),
                        ]
                    )
        val_dataset = CVDataset("validation", transform=None)
        test_dataset = CVDataset("testing", transform=None)
    else:
        train_dataset, val_dataset, test_dataset = get_cv_dataset(batch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # dataset = CVDataset("train")
    # dataset[0]
    # print(dataset[0][0].shape)
    train_dataset, val_dataset, test_dataset = get_cv_dataset(batch_size=32)
    print(test_dataset.shape)
    