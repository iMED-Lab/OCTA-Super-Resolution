from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)  
        # self.upsample = nn.Sequential(nn.Conv2d(3,1*4,3,padding=1),nn.PixelShuffle(2))
        # self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.upsample(x)
        # x = self.sigmoid(x)
        return x
