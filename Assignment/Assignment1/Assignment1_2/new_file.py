'''
Author: Xiang Pan
Date: 2021-09-30 17:25:27
LastEditTime: 2021-09-30 17:25:35
LastEditors: Xiang Pan
Description: 
FilePath: /Assignment1_2/new_file.py
xiangpan@nyu.edu
'''
data_jitter_brightness = transforms.Compose([
    # transforms.ColorJitter(brightness=-5),
    # transforms.ColorJitter(brightness=5),
    transforms.ColorJitter(brightness=5),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
    # transforms.ColorJitter(saturation=5),
    # transforms.ColorJitter(saturation=-5),
    transforms.ColorJitter(saturation=5),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
    # transforms.ColorJitter(contrast=5),
    # transforms.ColorJitter(contrast=-5),
    transforms.ColorJitter(contrast=5),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
    transforms.ColorJitter(hue=0.4),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
    transforms.RandomRotation(15),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
    transforms.RandomVerticalFlip(1),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
    transforms.RandomAffine(degrees=15,shear=2),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])
from randaugment import RandAugment, ImageNetPolicy

data_random_aug = transforms.Compose([
    RandAugment()
])