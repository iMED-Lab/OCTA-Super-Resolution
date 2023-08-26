import torch
from torch import nn

class one_conv(nn.Module):
    def __init__(self,inchanels,out_channels,kernel_size=3):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        convs = []
        for i in range(19):
            convs.append(one_conv(64, 64))
        self.conv = nn.Sequential(*convs)
    def forward(self, x):
        res = self.conv(x)
        res = self.conv1(res)
        res = torch.cat((x, res), 1)

        return res

class HARNet(nn.Module):
    def __init__(self):
        super(HARNet, self).__init__()
        num_block = 4
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.body = self.make_layer(ResBlock, num_block)
        self.body = ResBlock()
        self.upsample = nn.Sequential(nn.Conv2d(3,1*4,3,padding=1),nn.PixelShuffle(2))
    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        return nn.Sequential(*res_block)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.body(out)
        out = self.conv_4(out)
        out = self.body(out)
        out = self.conv_4(out)
        out = self.body(out)
        out = self.conv_4(out)
        out = self.body(out)
        # print(out.shape)
        out = self.conv_4(out)


        out = self.conv_3(out)
        # print(out.shape)
        out = torch.add(out, x)
        out = self.upsample(out)
        out = nn.Sigmoid()(out)


        return out





