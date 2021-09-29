'''
Author: Xiang Pan
Date: 2021-09-09 17:29:27
LastEditTime: 2021-09-29 17:09:16
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/nets.py
xiangpan@nyu.edu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision import models

class MY_CE(nn.Module):
    def __init__(self, eps=0.8, reduction="mean", pred_normalization = False): 
        super(MY_CE, self).__init__()
        self.reduction = reduction
        self.eps = eps 
        self.pred_normalization = pred_normalization   
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)   

    def forward(self, pred, target):
        # softmax first as CE-loss
        if self.pred_normalization:
            log_P = self.log_softmax(pred) - torch.log((self.softmax(pred)).sum())  
        else:
            log_P = self.log_softmax(pred) 
        class_num = pred.shape[1]
        one_hot = F.one_hot(target, num_classes=pred.shape[1]).to(pred.device)
        rev_hot = torch.ones(pred.shape[0], class_num).to(pred.device) - one_hot
        eps_hot = (1 - self.eps) * one_hot + (rev_hot * self.eps) / (pred.shape[1] - 1)
            
        
        loss = -eps_hot * log_P
        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss

def get_layer(model, layer_num):
    num = 0
    for i in model.children():
        if num == layer_num:
            break   
        num += 1
    return i

def get_layers_before(model, layer_before):
    layer_list = nn.ModuleList()
    num = 0
    for i in model.children():
        if i == layer_before:
            break    
        layer_list.append(i)
        num += 1
    return layer_list


# def get_layers_after
class ResNet12(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(ResNet12, self).__init__()
        self.model = models.resnet12(pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class vgg16(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(vgg16, self).__init__()
        self.model = models.vgg16_bn(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        logit = self.model(x)
        return logit

    def get_feature(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        return feature

    def forward_with_layer2(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        layer2_x = self.model.layer2(x)
        x = self.model.layer3(layer2_x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        logit = self.model.fc(feature)
        return layer2_x, logit
    
    def forward_from_layer2(self,layer2_x):
        x = self.model.layer3(layer2_x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        logit = self.model.fc(feature)
        return logit
    
    def forward_from_layer2_with_feature(self, layer2_x):
        x = self.model.layer3(layer2_x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        logit = self.model.fc(feature)
        return feature, logit


    def forward_with_feature(self, x):
        feature = self.get_feature(x)
        logit = self.model.fc(feature)
        return feature, logit
    
    def get_feature_layer(self):
        feature_layer = get_layers_before(self.model,self.model.fc)
        return feature_layer

    def get_feature(self, x):
        self.feature_layer = self.get_feature_layer()
        for i, l in enumerate(self.feature_layer):
            x = self.feature_layer[i](x) 
        feature = x
        feature = feature.squeeze()
        return feature

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, num_classes)
        self.feature_layer = self.get_feature_layer()
        # print(self.model)
        # print(self.feature_layer)

    def forward(self, x):
        logit = self.model(x)
        return logit

    # def get_feature(self,x):
    #     feature = self.feature_layer(x)
    #     return feature

    def forward_with_feature(self, x):
        feature = self.get_feature(x)
        logit = self.model.fc(feature)
        return feature, logit
    
    def get_feature_layer(self):
        feature_layer = get_layers_before(self.model,self.model.fc)
        return feature_layer

    def get_feature(self, x):
        self.feature_layer = self.get_feature_layer()
        for i, l in enumerate(self.feature_layer):
            x = self.feature_layer[i](x) 
        feature = x
        feature = feature.squeeze()
        return feature

class c_c_c(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        self.net
    
    def forward(self, x):
        return self.net(x)


class TfmNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TfmNet, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250*2*2, 350)
        self.fc2 = nn.Linear(350, num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.conv_drop(x)
        x = x.view(-1, 250*2*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        logit = self.fc2(x)
        # return F.log_softmax(x, dim=1)
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

        
class KaggleNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=43):
        super(KaggleNet, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=(5,5), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5), stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=(3,3), stride=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=512, out_features=num_classes),
        )
        
    def forward(self, x):
        feature = self.feature_extractor(x)
        logit = self.classifier(feature)
        return logit
    

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
        # probs = F.softmax(logits, dim=1)
        # return logits, probs
        return logits

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 43)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x 


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
            
        else:
            raise Exception("backbone name error")
    def forward(self, x):
        return self.backbone(x)
    def get_feature(self, x):
        return self.backbone.get_feature(x)
    def forward_with_feature(self, x):
        return self.backbone.forward_with_feature(x)
    def forward_with_layer2(self, x):
        return self.backbone.forward_with_layer2(x)
    def forward_from_layer2(self, layer2_x):
        return self.backbone.forward_from_layer2(layer2_x)
    def forward_from_layer2_with_feature(self, layer2_x):
        return self.backbone.forward_from_layer2_with_feature(layer2_x)



if __name__ == '__main__':
    net = NET()
    print(net)