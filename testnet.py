import torch
import torchvision.models as models
from torch import nn

import os
from transformer import interactive, interactive1_2

def mkdir(path):

    isExists = os.path.exists(path) # 判断路径是否存在，若存在则返回True，若不存在则返回False
    if not isExists: # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        return False


def decoder(input_channel, output_channel, num=4):
    if num == 4:
        decoder_body = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(output_channel, output_channel, kernel_size=(2, 2), stride=2)
                                     )
    elif num == 2:
        decoder_body = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(output_channel, output_channel, kernel_size=(2, 2), stride=2)
                                     )

    return decoder_body


def intergration(input_channel, output_channel):
    output = nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(2, 2), stride=2),
                           nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(output_channel, output_channel, kernel_size=(1, 1)),
                           nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                           nn.ReLU(inplace=True))
    return output

class VGG16_deconv(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16_deconv, self).__init__()
        vgg16bn_pretrained = models.vgg19_bn(pretrained=True)
        vgg16bn_pretrained_depth = models.vgg19_bn(pretrained=True)
        pool_list = [6, 13, 26, 39, 52]
        for index in pool_list:
            vgg16bn_pretrained.features[index].return_indices = True
            vgg16bn_pretrained_depth.features[index].return_indices = True

        self.encoder1 = vgg16bn_pretrained.features[:6]
        self.pool1 = vgg16bn_pretrained.features[6]
        self.dencoder1 = vgg16bn_pretrained_depth.features[:6]
        self.dpool1 = vgg16bn_pretrained_depth.features[6]
        self.trans1 = interactive1_2(d_model=128, ratio=2)

        self.encoder2 = vgg16bn_pretrained.features[7:13]
        self.pool2 = vgg16bn_pretrained.features[13]
        self.dencoder2 = vgg16bn_pretrained_depth.features[7:13]
        self.dpool2 = vgg16bn_pretrained_depth.features[13]
        self.trans2 = interactive1_2(d_model=256, ratio=2)

        self.encoder3 = vgg16bn_pretrained.features[14:26]
        self.pool3 = vgg16bn_pretrained.features[26]
        self.dencoder3 = vgg16bn_pretrained_depth.features[14:26]
        self.dpool3 = vgg16bn_pretrained_depth.features[26]
        self.trans3 = interactive(n=6, d_model=256, heads=4, dropout=0.1, activation="relu", pos_feats=32, num_pos_feats=128, ratio=2)

        self.encoder4 = vgg16bn_pretrained.features[27:39]
        self.pool4 = vgg16bn_pretrained.features[39]
        self.dencoder4 = vgg16bn_pretrained_depth.features[27:39]
        self.dpool4 = vgg16bn_pretrained_depth.features[39]
        self.trans4 = interactive(n=8, d_model=512, heads=4, dropout=0.1, activation="relu", pos_feats=16, num_pos_feats=256, ratio=2)


        self.encoder5 = vgg16bn_pretrained.features[40:52]
        self.pool5 = vgg16bn_pretrained.features[52]
        self.dencoder5 = vgg16bn_pretrained_depth.features[40:52]
        self.dpool5 = vgg16bn_pretrained_depth.features[52]
        self.trans5 = interactive(n=8, d_model=512, heads=4, dropout=0.1, activation="relu", pos_feats=8,
                                 num_pos_feats=256, ratio=2)

        self.decoder5 = decoder(1024, 512)

        self.decoder4 = decoder(1536, 256)

        self.decoder3 = decoder(768, 128)

        self.decoder2 = decoder(384, 64, 2)

        self.decoder1 = decoder(192, num_classes, 2)  # classes_num

    def forward(self, x, depthmap):
        encoder1 = self.encoder1(x)
        pool1, indices1 = self.pool1(encoder1)
        d1 = self.dencoder1(depthmap)
        d1, indicesd1 = self.dpool1(d1)
        out1, out1_1 = self.trans1(pool1, d1)

        d2 = self.dencoder2(d1)
        d2, indicesd2 = self.dpool2(d2)
        encoder2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(encoder2)
        out2, out2_2 = self.trans2(pool2, d2)

        d3 = self.dencoder3(d2)
        d3, indicesd3 = self.dpool3(d3)
        encoder3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(encoder3)
        out3, out3_3 = self.trans3(pool3, d3)

        d4 = self.dencoder4(d3)
        d4, indicesd4 = self.dpool4(d4)
        encoder4 = self.encoder4(pool3)
        pool4, indices4 = self.pool4(encoder4)
        out4, out4_4 = self.trans4(pool4, d4)

        d5 = self.dencoder5(d4)
        d5, indicesd5 = self.dpool5(d5)
        encoder5 = self.encoder5(pool4)
        pool5, indices5 = self.pool5(encoder5)
        out5, out5_5 = self.trans5(pool5, d5)
        decode = torch.cat((out5, out5_5), 1)

        decoder5 = self.decoder5(decode)
        decoder5 = torch.cat((decoder5, out4, out4_4), 1)
        decoder4 = self.decoder4(decoder5)
        decoder4 = torch.cat((decoder4, out3, out3_3), 1)
        decoder3 = self.decoder3(decoder4)

        decoder3 = torch.cat((decoder3, out2, out2_2), 1)
        decoder2 = self.decoder2(decoder3)

        decoder2 = torch.cat((decoder2, out1, out1_1), 1)
        decoder1 = self.decoder1(decoder2)

        return decoder1

class Imagemodel(nn.Module):
    def __init__(self):
        super(Imagemodel, self).__init__()
        self.model = VGG16_deconv()

    def forward(self, clip, depth_clip):
        return [self.model(frame, depth) for frame, depth in zip(clip, depth_clip)]
