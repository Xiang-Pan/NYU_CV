import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils import *
from randaugment import RandAugment


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
            self.X = torch.load(X_path).squeeze(1)
        else:
            raise ValueError("Invalid split")
        self.transform = transform
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        if self.split get_ipython().getoutput("= "testing":")
            if self.transform:
                return self.transform(self.X[idx]), self.y[idx]
            else:
                return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


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
                            CVDataset("train", transform=transforms.Compose([transforms.RandomHorizontalFlip(1),transforms.RandomVerticalFlip(1),])),
                            CVDataset("train", transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),])),
                            CVDataset("train", transform=transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),])),
                            CVDataset("train", transform=transforms.Compose([transforms.RandomAffine(degrees=15, shear=2),])),
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


import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False

parser = argparse.ArgumentParser()
parser.add_argument('--name',           type=str,    default=None,       required=False,)
parser.add_argument('--learning_rate',  type=float,  default=0.01,       required=False,)
parser.add_argument('--batch_size',     type=int,    default=32,         required=False,)
parser.add_argument('--warmup_epochs',  type=int,    default=-1,         required=False,)
parser.add_argument('--max_epochs',     type=int,    default=80,         required=False,)
parser.add_argument('--backbone_name',  type=str,    default="TfmNet",   required=False,)
parser.add_argument('--scheduler_name', type=str,    default="cosine",       required=False,)
parser.add_argument('--optimizer_name', type=str,    default="adam",     required=False,)
parser.add_argument('--weight_decay',   type=float,  default=1e-5,       required=False,)
parser.add_argument('--aug',            action='store_true',             default=True,)
parser.add_argument('--mixup',          action='store_true',             default=False,)
parser.add_argument('--label_smoothing',type=float,  default=False,)
parser.add_argument('--focal_loss',     action='store_true',             default=False,)
parser.add_argument('--seed',           type=int,             default=1)
parser.add_argument('--sweep_aug',      type=int,             default=0)

def get_option():
    option = parser.parse_args(args = [])
    return option
option = get_option()


option


# from nets import NET
import torch
import torch.nn as nn
import pickle
import pandas as pd
import torch.optim as optim
import os
import ttach as tta

# from task_datasets.cv_datasets import get_cv_dataloader
import wandb
import torch.nn.functional as F
import torchmetrics
from utils import *
from option import *
from warmup_scheduler import GradualWarmupScheduler


class NET(nn.Module):
    def __init__(self, backbone_name = "ResNet18", num_classes = 2, pretrained = False):
        super(NET, self).__init__()
        if backbone_name == "ResNet12":
            self.backbone = ResNet12(pretrained = pretrained, num_classes= num_classes)
        elif backbone_name == "ResNet18":
            self.backbone = ResNet18(pretrained = pretrained, num_classes= num_classes)
        elif backbone_name == "ResNet50":
            self.backbone = ResNet50(pretrained = pretrained, num_classes= num_classes)
        elif backbone_name == "LeNet5":
            self.backbone = LeNet5()
        elif backbone_name == "LaNet":
            self.backbone = LaNet()
        elif backbone_name == "SampleNet":
            self.backbone = SampleNet()
        elif backbone_name == "KaggleNet":
            self.backbone = KaggleNet()
        elif backbone_name == "EfficientNet":
            self.backbone = EfficientNet.from_name('efficientnet-b0', image_size=(32,32), num_classes=num_classes)
        elif backbone_name == "vgg16":
            self.backbone = vgg16(pretrained = pretrained, num_classes= num_classes)
        elif backbone_name == "inception_v3":
            self.backbone = models.inception_v3(pretrained = pretrained, aux_logits = True)
            self.backbone.AuxLogits.fc = nn.Linear(768, num_classes)
            self.backbone.fc = nn.Linear(2048, num_classes)
            print(self.backbone)
        elif backbone_name == "TfmNet":
            self.backbone = TfmNet(num_classes=num_classes)
        elif backbone_name == "TfmNetHighway":
            self.backbone = TfmNetHighway(num_classes=num_classes)
        else:
            raise Exception("backbone name error")
    def forward(self, x):
        return self.backbone(x)
    # def get_feature(self, x):
    #     return self.backbone.get_feature(x)
    # def forward_with_feature(self, x):
    #     return self.backbone.forward_with_feature(x)
    # def forward_with_layer2(self, x):
    #     return self.backbone.forward_with_layer2(x)
    # def forward_from_layer2(self, layer2_x):
    #     return self.backbone.forward_from_layer2(layer2_x)
    # def forward_from_layer2_with_feature(self, layer2_x):
    #     return self.backbone.forward_from_layer2_with_feature(layer2_x)


class LeNet5(nn.Module):
    def __init__(self, n_classes=43):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        logit = self.model(x)
        return logit


# Layer	Shape
# Input	32x32x3
# Convolution (valid, 5x5x6)	28x28x6
# Max Pooling (valid, 2x2)	14x14x6
# Activation (ReLU)	14x14x6
# Convolution (valid, 5x5x16)	10x10x16
# Max Pooling (valid, 2x2)	5x5x16
# Activation (ReLU)	5x5x16
# Flatten	400
# Dense	120
# Activation (ReLU)	120
# Dense	43
# Activation (Softmax)	43
class LaNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=43):
        super(LaNet, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=120, out_features=num_classes),
        )
        
    def forward(self, x):
        feature = self.feature_extractor(x)
        logit = self.classifier(feature)
        return logit


class TfmNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TfmNet, self).__init__()
        
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=5),
            nn.LeakyReLU(),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(100),
            nn.Dropout2d(0.5)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(100, 150, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(150),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(150),
            nn.Dropout2d(0.5)
        )
        
        self.c3 = nn.Sequential(
            nn.Conv2d(150, 250, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(250),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(250),
            nn.Dropout2d(0.5)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc1 = nn.Linear(1000, 350)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(350, num_classes)

    def stn1(self, x):
        xs = self.localization(x)
        # print(xs.shape, "xs") 
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        x = self.stn1(x)

        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        
        x = x.view(-1, 250 * 2 * 2)
        
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_dropout(x)
        
        logit = self.fc2(x)
        return logit


wandb.init(project="assignment1_2_submission")


import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from kornia.losses import focal_loss 

class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        print("focal_loss")

    def forward(self, input, target):
        return focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_scheduler(optimizer, args):
    scheduler_name = args.scheduler_name
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs-args.warmup_epochs)
    elif scheduler_name == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[7])
    elif scheduler_name == 'ReduceLROnPlateau':
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5,factor=0.5,verbose=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
    else:
        scheduler = None
    if scheduler is not None:
        if args.warmup_epochs > 0:
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler)
            return scheduler
    return scheduler


def train(max_epochs, model, optimizer, train_loader, val_loader, test_loader, args):
    
    max_train_acc = 0
    max_val_acc = 0

    optimizer.zero_grad()
    optimizer.step()
    if args.focal_loss:
        criterion = LossWrapper()
    elif args.label_smoothing > 0:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    scheduler = get_scheduler(optimizer, args)
    
    for cur_epoch in range(0, max_epochs):
        # scheduler_warmup.step(cur_epoch)
        # train
        model.train()
        train_acc_metric = torchmetrics.Accuracy().cuda()
        train_acc_metric.reset()
        train_loss_epoch = 0
        val_loss_epoch = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            train_loss = criterion(output, target)
            train_loss_epoch += train_loss.item()
            
            preds = output.softmax(dim=-1)
            acc = train_acc_metric(preds, target).cuda()
            
            wandb.log({'train_loss': train_loss})
            wandb.log({'train_acc': acc})

            if args.mixup:
                if cur_epoch > args.warmup_epochs:
                    mixed_x, y_a, y_b, lam = mixup_data(data, target)
                    mixed_pred = model(mixed_x)
                    mixup_loss = mixup_criterion(criterion, mixed_pred, y_a, y_b, lam)
                    train_loss += mixup_loss
                    wandb.log({'mixup_train_loss': mixup_loss.item()})
                
            
            train_loss.backward()
            optimizer.step()
            
        train_acc = train_acc_metric.compute()
        wandb.log({'train_acc_epoch': train_acc, "epoch": cur_epoch})
        wandb.log({'train_loss_epoch': train_loss_epoch/len(train_loader), "epoch": cur_epoch})
        max_train_acc = max(max_train_acc, train_acc)
        wandb.log({'max_train_acc': max_train_acc, "epoch": cur_epoch})

        # val
        model.eval()
        val_acc_metric = torchmetrics.Accuracy().cuda()
        val_acc_metric.reset()
        for batch_idx, (data, target) in enumerate(val_loader):
            
            data = data.cuda()
            target = target.cuda()
            output = model(data)

            val_loss = F.cross_entropy(output, target)
            val_loss_epoch += val_loss.item()
            preds = output.softmax(dim=-1)
            acc = val_acc_metric(preds, target).cuda()
            wandb.log({'val_loss': val_loss})
            wandb.log({'val_acc': acc})
        val_acc = val_acc_metric.compute()
        wandb.log({'val_acc_epoch': val_acc, "epoch": cur_epoch})
        wandb.log({'val_loss_epoch': val_loss_epoch/len(val_loader), "epoch": cur_epoch})
        max_val_acc = max(max_val_acc, val_acc)
        wandb.log({'mac_val_acc': max_val_acc, "epoch": cur_epoch})
        
        
        # predict test data
        path = "./outputs/"+wandb.run.id
        if not os.path.exists(path):
            os.system("mkdir -p get_ipython().run_line_magic("s"", " % path)")
        outfile_name = path+"/"+str(cur_epoch)+".csv"
        model_name = path+"/"+str(cur_epoch)+".pt"

        output_file = open(outfile_name, "w")
        
        # dataframe_dict = {"Filename" : [], "ClassId": []}
        df = pd.DataFrame(columns=['Filename', 'ClassId'])
        file_ids = pickle.load(open('./cached_datasets/testing/file_ids.pkl', 'rb'))
        
        for batch_idx, data in enumerate(test_loader):
    
            data = data.cuda()
            output = model(data)
            preds = torch.argmax(output.softmax(dim=-1), dim = -1).cpu().detach().tolist()
            
            file_id = file_ids[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            dft = pd.DataFrame(columns=['Filename', 'ClassId'], data=list(zip(file_id, preds)))
            df = df.append(dft, ignore_index=True)

        df.to_csv(outfile_name, index=False)
        print("Written to csv file {}".format(outfile_name))
        torch.save(model, model_name)
        
        
        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                # scheduler.step(val_acc)
                scheduler.step(np.around(val_acc.cpu().numpy(), decimals=2))
                wandb.log({'learning_rate': optimizer.param_groups[0]['lr'], "epoch": cur_epoch})
            else:
                scheduler.step(cur_epoch)
                wandb.log({'learning_rate': scheduler.get_last_lr(), "epoch": cur_epoch})
                


def get_auto_name(args):
    if args.scheduler_name is None:
        scheduler_name = 'None'
    else:
        scheduler_name = args.scheduler_name
    auto_name = '_'.join([  args.backbone_name,
                            scheduler_name,
                            str(args.learning_rate),
                            str(args.max_epochs),
                            str(args.batch_size),
                            str(args.warmup_epochs),
                            str(args.weight_decay),
                            str(args.label_smoothing),
                            args.optimizer_name
                        ])
    return auto_name


def set_args(args):
    # python main.py --backbone_name=TfmNet --batch_size=64 --learning_rate=0.0005 --max_epochs=100 --weight_decay=0 --scheduler=cosine --warmup_epochs=5 --aug --optimizer_name=adam --seed 1 --label_smoothing=0.01 --sweep_aug=1 --name=debug
    args.backbone_name = "TfmNet"
    args.batch_size = 64
    args.learning_rate = 0.005
    args.max_epochs = 100
    args.weight_decay = 0
    args.scheduler = "cosine"
    args.warmup_epochs = 5
    args.optimizer_name = "adam"
    args.seed = 1
    args.label_smoothing = 0.01
    args.sweep_aug = 1
    args.name = "submission"
    return args


def main():
    # args = get_option()
    args = set_args(option)
    torch.manual_seed(args.seed)
    if args.name is None:
        wandb.run.name = get_auto_name(args)
    else:
        wandb.run.name = args.name
    
    nclasses = 43 # GTSRB has 43 classes
    model = NET(backbone_name=args.backbone_name, num_classes=43, pretrained=False)
    model.apply(weight_init)
    # model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='tsharpen')
    model = model.cuda()
    if args.optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    # no_decay = list()
    # decay = list()
    # for m in model.modules():
    #     if isinstance(m, (nn.Linear, nn.Conv2d)):
    #         decay.append(m.weight)
    #         no_decay.append(m.bias)
    #     elif hasattr(m, 'weight'):
    #         no_decay.append(m.weight)
    #     elif hasattr(m, 'bias'):
    #         no_decay.append(m.bias)

    if args.sweep_aug == 1:
        args.aug = True
    else:
        args.aug = False
    train_dataloader, val_dataloader, test_dataloader = get_cv_dataloader(batch_size = args.batch_size, augument=args.aug)
    train(args.max_epochs, model, optimizer, train_dataloader, val_dataloader, test_dataloader, args)


# from nets import NET
import torch
import torch.nn as nn
import pickle
import pandas as pd
import torch.optim as optim
import os
import ttach as tta
# from task_datasets.cv_datasets import get_cv_dataloader
import wandb
import torch.nn.functional as F
import torchmetrics
from utils import *
from option import *
from warmup_scheduler import GradualWarmupScheduler
main()


api = wandb.Api()
team, project, run_id = "xiang-pan", "assignment1_2_submission", "2zdy4mui"
run = api.run(f"{team}/{project}/{run_id}")
metrics_dataframe = run.history()

display()  # you may need to zoom out to see the whole window!


get_ipython().run_cell_magic("html", "", """<iframe src="https://wandb.ai/xiang-pan/assignment1_2_submission/runs/7teg2ng5?workspace=user-xiang-pan" width="1200" height="1000"></iframe>""")


import torchvision.models as models
from torchsummary import summary

model = TfmNet().cuda()
 
summary(model, (3, 32, 32))
# -1 represent the batch size here
