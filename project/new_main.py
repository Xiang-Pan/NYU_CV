'''
Author: Xiang Pan
Date: 2021-12-13 20:23:54
LastEditTime: 2021-12-13 20:23:56
LastEditors: Xiang Pan
Description: 
FilePath: /project/new_main.py
@email: xiangpan@nyu.edu
'''
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)