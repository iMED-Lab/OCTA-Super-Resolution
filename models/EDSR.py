import torch
from torch import nn


n_feat = 128
kernel_size = 3

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class _Res_Block(nn.Module):
    def __init__(self):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        y = self.relu(self.res_conv(x))
        y = self.res_conv(y)
        y *= 0.1
        y = torch.add(y, x)
        return y


class edsr(nn.Module):
    def __init__(self):
        super(edsr, self).__init__()


        in_ch = 3
        num_blocks = 8

        self.conv1 = nn.Conv2d(in_ch, n_feat, kernel_size, padding=1)
        self.conv_up = nn.Conv2d(n_feat, n_feat*4, kernel_size, padding=1)
        self.conv_out = nn.Conv2d(n_feat, in_ch, kernel_size, padding=1)

        self.body = self.make_layer(_Res_Block, num_blocks)

        self.upsample = nn.Sequential(self.conv_up,nn.PixelShuffle(2))

    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        return nn.Sequential(*res_block)

    def forward(self, x):



        out = self.conv1(x)
        res = self.body(out)
        res = out + res
        out = self.upsample(res)
        out = self.conv_out(out)
        out = nn.Sigmoid()(out)
        # print(out)
        return out
