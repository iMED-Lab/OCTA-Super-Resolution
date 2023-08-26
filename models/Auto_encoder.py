# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.up(x)

        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3):
        super(U_Net, self).__init__()
        # self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)

        x3 = self.Conv3(x2)

        x4 = self.Conv4(x3)

        return x4
class ResUNet1(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(ResUNet1, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # self.vgg = U_Net(img_ch=3)
        # self.xception = Inception3()
        #self.encoder = Encoder()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        # self.Up5 = up_conv(ch_in=512, ch_out=256)
        #self.Up_conv5 = res_conv_block(ch_in=256, ch_out=256)
        self.mean_head = nn.Sequential(nn.Conv2d(128, 128, 1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(0.2,inplace=True))
        self.logvar_head = nn.Sequential(nn.Conv2d(128, 128, 1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(0.2,inplace=True))


        self.Up4 = up_conv(ch_in=256, ch_out=128)
        # #self.Up_conv4 = res_conv_block(ch_in=128, ch_out=128)
        #
        self.Up3 = up_conv(ch_in=128, ch_out=64)
        # #self.Up_conv3 = res_conv_block(ch_in=64, ch_out=64)
        #
        self.Up2 = up_conv(ch_in=64, ch_out=64)
        # #self.Up_conv2 = res_conv_block(ch_in=64, ch_out=64)
        #
        self.Up1 = up_conv(ch_in=64, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)


    def forward(self, x):
        # encoding path
        x0_ = torch.cat([x,x,x],1)
        x0 = self.firstconv(x0_)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        # mu0 = self.mean_head(x3)
        # logvar0 = self.logvar_head(x3)
        # mu3 = self.mean_head2(x4_2)
        # logvar3 = self.logvar_head2(x4_2)
        # mu_lst = [mu0]
        # logvar_lst = [logvar0]



        #
        # if self.training:
        #    std = torch.exp(0.5*logvar0)
        #    eps = torch.randn_like(std)
        #    z = mu0 + std * eps
        # else:
        #    z = mu0

           # encoding path
        d4 = self.Up4(x4)


        d3 = self.Up3(x3)

        d2 = self.Up2(d3)


        d1 = self.Up1(d2)


        d0 = self.Conv_1x1(d1)
        d = nn.Sigmoid()(d0)

        return  d