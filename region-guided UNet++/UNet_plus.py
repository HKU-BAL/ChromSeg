import torch
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models

class DoubleConv(nn.Module):
    # The convolutional layer: conv3-relu-conv3-relu
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UNet_plus2(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet_plus2, self).__init__()
        self.n_channels = in_ch
        self.n_classes = out_ch
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        # Encoder: U-Net
        #self.conv0_0 = nn.Sequential(
        #    nn.Conv2d(in_ch, filters[0], kernel_size=7, stride=1, padding=3,
        #         bias=False),
        #    nn.BatchNorm2d(filters[0]),
        #    nn.ReLU(inplace=True)
        #)
        self.conv0_0 = DoubleConv(self.n_channels, filters[0], filters[0])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.conv1_0= DoubleConv(filters[0], filters[1], filters[1])
        self.conv2_0= DoubleConv(filters[1], filters[2], filters[2])
        self.conv3_0= DoubleConv(filters[2], filters[3], filters[3])

        # Upsample layer(Deconv)
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)

        # Mid Layer
        self.conv0_1 = DoubleConv(filters[0]*2, filters[0], filters[0]) 
        self.conv1_1 = DoubleConv(filters[1]*2, filters[1], filters[1])
        self.conv2_1 = DoubleConv(filters[2]*2, filters[2], filters[2])

        self.conv0_2 = DoubleConv(filters[0]*2, filters[0], filters[0])
        self.conv1_2 = DoubleConv(filters[1]*2, filters[1], filters[1])

        self.conv0_3 = DoubleConv(filters[0]*2, filters[0], filters[0])

        # attention
        self.attention0 = ContourAttention(filters[0])
        self.attention1 = ContourAttention(filters[1])
        
        self.contour = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], out_ch, 1)
        )
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], out_ch, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        p = self.pool(x0_0)
        x1_0 = self.conv1_0(p)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1)) + x0_0
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        
        x0_2 = self.conv0_2(torch.cat([x0_1, self.up1_1(x1_1)], 1)) + x0_1
        # x1_2 = self.conv1_2(torch.cat([self.attention1(x1_1, x1_0), self.up2_1(x2_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up2_1(x2_1)], 1))
        
        # x0_3 = self.conv0_3(torch.cat([self.attention0(x0_2, x0_0), self.up1_2(x1_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up1_2(x1_2)], 1))
        
        contour = self.contour(torch.cat([x0_1, x0_2], dim=1))
        output = self.final(x0_3)

        return self.sigmoid(output), self.sigmoid(contour)