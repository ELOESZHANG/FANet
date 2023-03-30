#!/usr/bin/env python
# coding: utf-8
#
# Author: speedinghzl
# URL: https://github.com/speedinghzl/pytorch-segmentation-toolbox

import time
import logging

import torch
import torch.nn as nn

from .logger import get_logger



logger = get_logger()

"""def init_layer(key):
        #初始化层，（未进行预训练并微调）
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001"""

def weight_init(self):
    if isinstance(self,nn.Conv3d):
        #nn.init.kaiming_normal_(self.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.weight.data,gain=0.01)
        #nn.init.normal_(self.weight.data,mean=0,std=0.01)
        nn.init.constant_(self.bias.data,0)
    elif isinstance(self,nn.BatchNorm2d):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'state_dict' in state_dict.keys():
           state_dict = state_dict['state_dict']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if not is_restore:
        # extend the input channels of FGPLG from 3 to 7
        v2 = model.backbone.resnet.conv1.weight
        if v2.size(1) > 3:
            v = state_dict['backbone.resnet.conv1.weight']
            v = torch.cat((v,v2[:,3:,:,:]), dim=1)
            state_dict['backbone.resnet.conv1.weight'] = v

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys


    for k in missing_keys:
        weight_init(k)


    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model