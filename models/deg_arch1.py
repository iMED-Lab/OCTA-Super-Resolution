import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from other_models import U_Net


class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)

class RDB(nn.Module):
    def __init__(self,G0=64,C=6,G=32,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        #local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)
    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x

class ResBlock(nn.Module):
    def __init__(self, nf, ksize, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        
        self.nf = nf
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize//2),
            norm(nf), act(),
            nn.Conv2d(nf, nf, ksize, 1, ksize//2)
        )
    
    def forward(self, x):
        return torch.add(x, self.body(x))

class Quantization(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.n = n

    def forward(self, inp):
        out = inp * 255.0
        flag = -1
        for i in range(1, self.n + 1):
            out = out + flag / np.pi / i * torch.sin(2 * i * np.pi * inp * 255.0)
            flag = flag * (-1)
        return out / 255.0

class KernelModel(nn.Module):
    def __init__(self, scale):
        super().__init__()

        # self.opt = opt
        self.scale = scale


        nc, nf, nb = 64, 64, 8
        self.nc = nc
        ksize = 21

        # if opt["spatial"]:
        head_k = 3
        body_k = 3

        # else:
        #     head_k = body_k = 1
        
        # if opt["mix"]:
        in_nc = nc + 1
        # else:
        #     in_nc = nc

        deg_kernel = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            *[
                # ResBlock(nf=nf, ksize=body_k)
                ResBlock(nf=nf, ksize=body_k)
                for _ in range(nb)
                ],
            nn.Conv2d(nf, ksize ** 2, 1, 1, 0),
            nn.Softmax(1)
        ]

        self.deg_kernel = nn.Sequential(*deg_kernel)
        # self.deg_kernel = deg_kernel()
        # if opt["zero_init"]:
        nn.init.constant_(self.deg_kernel[-2].weight, 0)
        nn.init.constant_(self.deg_kernel[-2].bias, 0)
        self.deg_kernel[-2].bias.data[ksize ** 2 // 2] = 1
        self.pad = nn.ReflectionPad2d(ksize//2)
        self.downsample = nn.AvgPool2d(2)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = H // self.scale
        w = W // self.scale

        # if self.opt["nc"] > 0:
        # if self.opt["spatial"]:

        zk = torch.randn(B, self.nc, H, W).to(x.device)
        # else:
        #     zk = torch.randn(B, self.opt["nc"], 1, 1).to(x.device)
        #     if self.opt["mix"]:
        #         zk = zk.repeat(1, 1, H, W)
        #
        # if self.opt["mix"]:
        #     if self.opt["nc"] > 0:
        #         inp = torch.cat([x, zk], 1)
        #     else:
        #         inp = x
        # else:
        inp = torch.cat([x, zk], 1)
        
        ksize = 21
        kernel = self.deg_kernel(inp).view(B, 1, ksize**2, *inp.shape[2:])

        x = x.view(B*C, 1, H, W)
        x = F.unfold(
            self.pad(x), kernel_size=ksize, stride=1, padding=0).view(B, C, ksize**2, H, W)


        x1 = torch.mul(x, kernel).sum(2).view(B, C, H, W)
        x1 = self.downsample(x1)
        # kernel = kernel.view(B, ksize, ksize, *inp.shape[2:]).squeeze()

        return x1

class NoiseModel(nn.Module):
    def __init__(self,  scale):
        super().__init__()

        self.scale = scale
        nc, nf, nb = 64, 64, 8
        self.nc = nc
        ksize = 21

        # if opt["spatial"]:
        head_k = 3
        body_k = 3

        # if opt["spatial"]:
        #     head_k = opt["head_k"]
        #     body_k = opt["body_k"]
        # else:
        #     head_k = body_k = 1
        #
        # if opt["mix"]:
        in_nc = nc
        # else:
        # in_nc = nc

        deg_noise = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            *[
                ResBlock(nf=nf, ksize=body_k)
                for _ in range(nb)
                ],
            nn.Conv2d(nf, 1, 1, 1, 0),
        ]
        self.deg_noise = nn.Sequential(*deg_noise)

        # if opt["zero_init"]:
        nn.init.constant_(self.deg_noise[-1].weight, 0)
        nn.init.constant_(self.deg_noise[-1].bias, 0)
        # else:
        #     nn.init.normal_(self.deg_noise[-1].weight, 0.001)
        #     nn.init.constant_(self.deg_noise[-1].bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # if self.opt["nc"] > 0:
        #     if self.opt["spatial"]:
        zn = ((torch.randn([B, 1, H, W]))*0.05 + 0.0).to(x.device)
            # else:
            #     zn = torch.randn(x.shape[0], self.opt["nc"], 1, 1).to(x.device)
            #     if self.opt["mix"]:
            #         zn = zn.repeat(1, 1, H, W)
        
        # if self.opt["mix"]:
        #     if self.opt["nc"] > 0:
        inp = zn
        #     else:
        #         inp = x
        # else:
        # inp = zn
        # noise = self.deg_noise(inp)
        noise = inp
        return noise

class DegModel(nn.Module):
    def __init__(
        self,  scale=2, nc_img=3, kernel_opt=True, noise_opt=True):
        super().__init__()

        self.scale = scale

        self.kernel_opt = kernel_opt
        self.noise_opt = noise_opt

        if kernel_opt is not None:
            self.deg_kernel = KernelModel(scale)
        #
        if noise_opt is not None:
           self.deg_noise = NoiseModel(scale)

        # else:
        self.quant = Quantization()
        
    def forward(self, inp):
        B, C, H, W = inp.shape
        h = H // self.scale
        w = W // self.scale

        # kernel
        if self.kernel_opt is not None:
            x = self.deg_kernel(inp)
        # else:
        #     x = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
        #     kernel = None

        # noise
        if self.noise_opt is not None:
            noise = self.deg_noise(x)
            x = self.quant(x)
        else:
            noise = None
            x = self.quant(x)
        return x, noise

