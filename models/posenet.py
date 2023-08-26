import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
import models.RDN



class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, pretrain = False, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pretrain = pretrain
        # self.pre = models.RDN.rdn().cuda()

        self.pre1 = nn.Sequential(
            Residual(inp_dim=2, out_dim=inp_dim)
        )
        
        self.hgs = nn.ModuleList([
        nn.Sequential(
            Hourglass(3, inp_dim, bn, increase),
        ) for i in range(nstack)])
        
        self.features = nn.ModuleList([
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)])
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)])
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)])
        self.nstack = nstack
        self.upsample = nn.Sequential(nn.Conv2d(1, 1 * 4, 3, padding=1), nn.PixelShuffle(2))
        # self.conv = nn.Conv2d(oup_dim,1,3,padding=1)

        self.sigmoid = nn.Sigmoid()
        # self.agg = Conv(1, 1, 1, relu=False, bn=False)

    def forward(self, imgs):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        # image = imgs
        # x = self.pre(imgs)
        combined_hm_preds = []
        # x_ = torch.cat([imgs,imgs,imgs,imgs],dim=1)
        # imgs1 = self.upsample(imgs1)
        x_ = torch.cat([imgs,imgs],dim=1)

        # print(x.shape)
        x_ = self.pre1(x_)
        for i in range(self.nstack):
            hg = self.hgs[i](x_)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            if i < self.nstack - 1:
                x_ = x_ + self.merge_preds[i](preds) + self.merge_features[i](feature)

        # preds = self.sigmoid(self.upsample(preds))

            # combined_hm_preds.append(preds)

        return self.sigmoid(preds)

