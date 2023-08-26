#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-22 09:54
# * Last modified : 2018-08-22 09:54
# * Filename      : RDN.py
# * Description   : this part for us is realize the RDN model from the paper
# * all detail you can see from the paper "Residual Dense Network for Image SR"
# **********************************************************

from BasicModule import *
import torch.nn as nn
import torch
import time

class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), 1)

class adjust_model(nn.Module):
    def __init__(self, inchanels, growth_rate):
        super(adjust_model,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchanels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(growth_rate, growth_rate, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)


class RDB(nn.Module):
    def __init__(self, G0=64, C=6, G=32, kernel_size=3):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0 + i * G, G))
        self.conv = nn.Sequential(*convs)
        # local_feature_fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, kernel_size=1, padding=0, stride=1)
        # self.new_model = adjust_model(G0,G0)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        # lff = self.new_model(lff)
        # local residual learning
        return lff + x


class rdn(basic):  # BasicModule.##
    def __init__(self):
        '''
        opts: the system para
        '''
        super(rdn, self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64filter
        '''
        self.D = 20
        self.C = 6
        self.G = 32
        self.G0 = 64
        kernel_size = 3
        input_channels = 3
        # shallow feature extraction
        self.SFE1 = nn.Conv2d(input_channels, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.SFE2 = nn.Conv2d(self.G0, self.G0, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        # RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0, self.C, self.G, kernel_size))
        # Global feature fusion
        self.GFF = nn.Sequential(
            nn.Conv2d(self.D * self.G0, self.G0, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kernel_size, padding=kernel_size >> 1, stride=1),
        )
        # upsample net
        self.up_net = nn.Sequential(
            nn.Conv2d(self.G0, self.G, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
            nn.Conv2d(self.G, 3, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1),
        )
        # self.upsample1 = nn.Conv2d(3,1,3,padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(3, 1 * 4, 3, padding=1), nn.PixelShuffle(2))
        # self.conv3 = nn.Conv2d(64,1,3,padding=1)
        # init

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # f-1
        f__1 = self.SFE1(x)
        out = self.SFE2(f__1)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs, 1)
        out = self.GFF(out)
        out = f__1 + out
        out = self.up_net(out)
        # print(out.size())
        out = self.upsample(out)
        out = self.sigmoid(out)

        return out
