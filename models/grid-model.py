import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18,resnet34,resnet50,resnet101,vgg16
class DeepHomographyModel(nn.Module):
    def __init__(self):
        super(DeepHomographyModel, self).__init__()
        self.net = resnet50(pretrained=False)
        self.net.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # # self.attention = attention_net()
        self.net.fc = nn.Linear(512,6)
        self.tanh = nn.Tanh()
    def forward(self, x,y):
        z = torch.cat([x,y],dim=1)
        out = self.net(z)
        print(out)
        grids0 = F.affine_grid(out.reshape(out.shape[0], 2, 3), x.shape)
        warped_image_m_1 = F.grid_sample(x, grids0)
        # warped_image_m_1 = warped_image_m_1.permute(0, 2, 1, 3)
        # out = pi * torch.tanh(out)
        # out = torch.cat([out[:, 1:], -pi * torch.tanh(out[:, 1:])], 1)
        # print(out.shape)
        return out, warped_image_m_1

