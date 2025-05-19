from .unet_parts import *
# from .CBAM import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.se_block1 = SELayer(channel=2048)
        # self.se_block = SELayer(channel=1024)
        
        # self.CBAM1 = CBAM(64)
        # self.CBAM1 = CBAM(128)
        # self.CBAM2 = CBAM(256)
        # self.CBAM3 = CBAM(512)
        # self.CBAM4 = CBAM(1024)

        self.sa1 = sa_layer(128)
        self.sa2 = sa_layer(256)
        self.sa3 = sa_layer(512)
        self.sa4 = sa_layer(1024)

        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512) 
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 下采样
        # input_channels = x.size(1)
        # # 动态定义self.inc
        # self.inc = DoubleConv(input_channels, 64)
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        
        x2 = self.sa1(x2)
        x3 = self.sa2(x3)
        x4 = self.sa3(x4)
        x5 = self.sa4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits