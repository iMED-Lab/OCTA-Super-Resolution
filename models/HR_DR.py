import torch
import torch.nn as nn

class H_D(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(H_D,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)