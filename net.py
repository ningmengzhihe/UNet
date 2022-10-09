import torch
from torch import nn
from torch.nn import functional as F
# torchvision是独立于pytorch的关于图像操作的一些方便工具库

# Unet网络结构组成模块：（1）conv_block包含2个conv（2）up sampling（3）down sampling
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            # 3*3的卷积，步长stride为1，padding为1，偏移量bias设置成False因为要使用BatchNorm
            # padding_mode='reflect’ 就是对图片进行镜像操作，填充区域是对原始图片的边缘区域进行镜像；镜像填充，最后一个像素不镜像
            # 如果用0做padding那么0没有特征信息，reflect来padding的都是有特征的
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        '''
        DownSample不改变channel，只改变h和w，所以out_channel=in_channel
        Note:下采样不是池化
        :param ch:
        '''
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel), # 可以用也可以不用batchnorm
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # 插值之后有一个卷积，1*1，步长是1，只是为了降通道
        self.layer = nn.Conv2d(channel, channel//2, 1, 1)

    def forward(self, x, feature_map):
        '''

        :param x:
        :param feature_map: 之前的特征图
        :return:
        '''
        # 转置卷积（图片周围填充一些空洞让图片变大）产生很多空洞对图像分割影响有些大，这里用另一种方法——插值法
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 变为原来的2倍
        out = self.layer(up)
        # concat,因为维度是(N,C,H,W)所以在dim=1也就是C这个维度上concat
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)  # 没有看懂这里为什么是1024不是512，可能和concat有关
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)  # 3个通道代表RGB彩色
        self.Th = nn.Sigmoid() # 分割只需要区分0（无颜色）和1（彩色）

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))  # 这里要concat
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)  # 用来测试，定义2个图片
    net = UNet()
    print(net(x).shape)
    # 输出torch.Size([2, 3, 256, 256])说明没有问题





