import torch
from torch import nn
import models.drln
from  models import HARNET,RDN,srcnn,EDSR,drln

from models.dynamic_conv import Dynamic_conv2d
from models.posenet import PoseNet


def conv1X1(inchanels, outchanels, kernel_size):
    return Dynamic_conv2d(in_planes=inchanels, out_planes=outchanels,
                          kernel_size=kernel_size, padding=kernel_size >> 1, K=inchanels, groups=inchanels)


def conv3X3(inchanels, outchanels, kernel_size):
    return Dynamic_conv2d(in_planes=inchanels, out_planes=outchanels,
                          kernel_size=kernel_size, padding=kernel_size >> 1, K=inchanels, groups=inchanels)


def conv5X5(inchanels, outchanels, kernel_size):
    return Dynamic_conv2d(in_planes=inchanels, out_planes=outchanels,
                          kernel_size=kernel_size, padding=kernel_size >> 1, K=inchanels, groups=inchanels)


class MDA(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(MDA, self).__init__()
        # self.conv0 = nn.Conv2d(inplanes,inplanes*8,kernel_size=3, padding = 1,stride= 1)

        self.conv1 = conv1X1(inplanes, inplanes*2, kernel_size=1)
        self.conv2 = conv3X3(inplanes*2, inplanes*2, kernel_size=3)
        self.conv3 = conv5X5(inplanes*2, inplanes*2, kernel_size=5)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes*16, planes, kernel_size=3, padding=1, stride=1)
        self.upsample = nn.Sequential(nn.Conv2d(8, 1 * 4, 3, padding=1), nn.PixelShuffle(2))
        self.tanh = nn.Sigmoid()

    def forward(self, x1,x2):
        x = torch.cat([x1,x2],dim=1)
        # x = (self.conv0(x))
        x = (self.conv1(x))

        out4 = self.tanh(self.upsample(x))

        return out4



class Agg(nn.Module):
    def __init__(self,  pretrain=False, **kwargs):
        super(Agg, self).__init__()

        self.pretrain = pretrain
        if self.pretrain == True:
            model = torch.load(
                '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/model/haohy_domain_ada_sr_rose_rdn/G-99.pth')
            self.pre = model.module
            # for param in self.pre.parameters():
            #     param.requires_grad = False
        else:
            self.pre = drln.DRLN().cuda()

        if self.pretrain == True:
            model = torch.load(
                '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/model/haohy_domain_ada_sr_rose_pose/G-167.pth')
            self.pre1 = model.module
            for param in self.pre1.parameters():
                param.requires_grad = False
        # else:
        #     self.pre1 = PoseNet(nstack=4, inp_dim=32, oup_dim=1).cuda()
        self.MDA = MDA(inplanes=4,planes=1)
        self.MDA1 = PoseNet(nstack=1, inp_dim=32, oup_dim=1).cuda()


        self.sigmoid = nn.Sigmoid()


    def forward(self, imgs):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        out1 = self.pre(imgs)
        # print(x.shape)
        # x_ = self.pre1(imgs)

        # out2 = self.MDA1(imgs)
        # # out = torch.cat([out1,out2],dim=1)
        # out = self.MDA(out1,out2)



        return out1

