#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

import random

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils import data
import numpy as np

def rgbd_transforms(image_mode, input_size,output_size, mean=[0.511, 0.619, 0.532], std=[0.241, 0.236, 0.244],
                    mean1=[0.341, 0.360, 0.753], std1=[0.208, 0.269, 0.241]):#mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    #数据增强 代码中image_mode=false没有用到 主要只用到totensor
    data_transforms = {
        'train': transforms.Compose([
            ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3, image_mode=image_mode),
            RandomResizedCrop(input_size, image_mode),
            RandomFlip(image_mode),
            ToTensor(),
            Normalize(mean=mean,
                    std=std, mean1=mean1, std1=std1)
        ]) if image_mode else transforms.Compose([
                                Resize(input_size,
                                output_size),
                                ToTensor(),
                                Normalize(mean=mean,
                                        std=std, mean1=mean1, std1=std1)
        ]),
        'val': transforms.Compose([
            Resize(input_size,output_size),
            ToTensor(),
            Normalize(mean=mean,
                            std=std, mean1=mean1, std1=std1)
        ]),
        'test': transforms.Compose([
            Resize(input_size,output_size),
            ToTensor(),
            Normalize(mean=mean,
                            std=std, mean1=mean1, std1=std1)
        ]),
    }
    return data_transforms

class ColorJitter(transforms.ColorJitter):
    def __init__(self, image_mode, **kwargs):
        super(ColorJitter, self).__init__(**kwargs)
        self.transform = None
        self.image_mode = image_mode
    def __call__(self, sample):
        if self.transform is None or self.image_mode:
            self.transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
        sample['image'] = self.transform(sample['image'])
        return sample

class RandomResizedCrop(object):
    """
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, image_mode, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.i, self.j, self.h, self.w = None, None, None, None
        self.image_mode = image_mode
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.i is None or self.image_mode:
            self.i, self.j, self.h, self.w = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, self.i, self.j, self.h, self.w, self.size, Image.BILINEAR)
        label = F.resized_crop(label, self.i, self.j, self.h, self.w, self.size, Image.BILINEAR)
        sample['image'], sample['label'] = image, label
        return sample

class RandomFlip(object):
    """Horizontally flip（水平翻转） the given PIL Image randomly with a given probability.
    """
    def __init__(self, image_mode):
        self.rand_flip_index = None
        self.image_mode = image_mode
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.rand_flip_index is None or self.image_mode:
            self.rand_flip_index = random.randint(-1,2)
        # 0: horizontal flip, 1: vertical flip, -1: horizontal and vertical flip
        if self.rand_flip_index == 0:
            image = F.hflip(image)
            label = F.hflip(label)
        elif self.rand_flip_index == 1:
            image = F.vflip(image)
            label = F.vflip(label)
        elif self.rand_flip_index == 2:
            image = F.vflip(F.hflip(image))
            label = F.vflip(F.hflip(label))
        sample['image'], sample['label'] = image, label
        return sample

class Resize(object):
    """ Resize PIL image use both for training and inference"""
    def __init__(self, input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, sample):
        image,label, depth=sample['image'],  sample['label'],sample['depth']# =
        image = F.resize(image, self.input_size, F.InterpolationMode.BILINEAR)
        depth = F.resize(depth, self.input_size, F.InterpolationMode.BILINEAR)
        if label is not None:
            label = F.resize(label, self.output_size, F.InterpolationMode.BILINEAR)
        sample['image'],sample['label'],sample['depth'] =image, label, depth#=  
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        label,depth ,image= sample['label'],sample['depth'], sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # Image range from [0~255] to [0.0 ~ 1.0]
        image = F.to_tensor(image)
        depth = F.to_tensor(depth)
        if label is not None:
            label = torch.from_numpy(np.array(label)).unsqueeze(0)#.float()
        return {'image': image,'label': label,'depth':depth} #{}

class Normalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) – Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """
    # default caffe mode
    def __init__(self, mean, std, mean1, std1):
        self.mean = mean
        self.std = std
        self.mean1 = mean1
        self.std1 = std1

    def __call__(self, sample):
        image,label, depth=sample['image'], sample['label'],sample['depth']#= 
        image = F.normalize(image, self.mean, self.std)
        #depth = torch.nn.functional.normalize(depth, dim=2)#F.normalize
        # depth = (depth-0.5)/0.5
        # depth = (depth - self.mean1) / self.std1
        depth = F.normalize(depth, self.mean1, self.std1)
        #depth = depth[0,:]
        #label 不進行歸一化，只改變大小和張量類型
        return {'image': image,'label':label, 'depth':depth}#

