import torch as t
import torch.nn as nn
from einops import rearrange
from model.encoder import ViT
from torch.nn import functional as F


class restruct(nn.Module):
    def __init__(self, in_channel):
        super(restruct, self).__init__()
        self.r_layer = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_channel // 32, 1, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x = self.r_layer(x)
        return x


class gen_Block(nn.Module):
    def __init__(self, in_channel, a):
        super(gen_Block, self).__init__()
        if a == 0:
            self.c_layer = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // 2, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(in_channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel // 2, in_channel // 8, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(in_channel // 8),
                nn.ReLU(inplace=True),
            )
        else:
            self.c_layer = nn.Sequential(
                nn.Conv2d(in_channel, 24, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 6, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(6),
                nn.ReLU(inplace=True),
                nn.Conv2d(6, 1, 3, 1, 1, bias=False),
            )

    def forward(self, x):
        x = self.c_layer(x)
        return x


class Contour(nn.Module):
    def __init__(self, ch_in):
        super(Contour, self).__init__()
        self.Conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        self.IN = nn.InstanceNorm2d(ch_in)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.Conv(x)
        x1 = self.IN(x1)
        x1 = self.ReLU(x1)
        return x1


class Contour_up(nn.Module):
    def __init__(self, ch_in):
        super(Contour_up, self).__init__()
        self.Conv = nn.ConvTranspose2d(ch_in, ch_in-32, kernel_size=4, stride=2, padding=1)
        self.IN = nn.InstanceNorm2d(ch_in-32)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv(x)
        x = self.IN(x)
        x = self.ReLU(x)
        return x


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out=32, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = Recurrent_block(ch_out, t=t)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.IN = nn.InstanceNorm2d(ch_out)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.Conv_1x1(x)
        x0 = self.IN(x0)
        x0 = self.ReLU(x0)
        x1 = self.RCNN(x0)
        x1 = x1 + x0
        x2 = self.RCNN(x1)
        return t.cat((x, x0 + x2), dim=1)


class RRCNN_dblock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_dblock, self).__init__()
        self.RCNN = Recurrent_block(ch_out, t=t)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.IN = nn.InstanceNorm2d(ch_out)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.IN(x)
        x = self.ReLU(x)
        x1 = self.RCNN(x)
        x1 = x1 + x
        x2 = self.RCNN(x1)
        return x + x2


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.IN = nn.InstanceNorm2d(ch_out)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
                x1 = self.IN(x1)
                x1 = self.ReLU(x1)
            x1 = self.conv(x + x1)
            x1 = self.IN(x + x1)
            x1 = self.ReLU(x + x1)
        return x1


class CBAM(nn.Module):
    def __init__(self, c1, r=16, d=1):
        super(CBAM, self).__init__()
        c_ = int(c1 // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c_, c1, kernel_size=1)
        )
        self.res_conv = nn.Conv2d(c1, c1//2, 3, padding=1)
        self.res_conv1 = nn.Conv2d(c1, c1//2+d, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=7 // 2)
        self.conv2 = nn.Conv2d(c1//2, c1, 3, padding=1)

    def forward(self, y):
        ca = t.cat([
            F.adaptive_avg_pool2d(y, 1),
            F.adaptive_max_pool2d(y, 1)
        ], dim=3)
        ca = ca.sum(dim=3, keepdims=True)
        ca1 = self.mlp(ca)
        ca = t.add(ca, ca1)
        ca = t.sigmoid(ca)
        y1 = ca * y
        return y1


class DenseDownsample(nn.Module):
    def __init__(self, ch_in, grate=32):
        super(DenseDownsample, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(ch_in, grate, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(grate),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return t.cat((self.single_conv(x), self.pooling(x)), dim=1)


class DRA_UNet(nn.Module):
    def __init__(self, inc=3, n_classes=1, grate=32, base_chns=16):
        super(DRA_UNet, self).__init__()
        self.conv1_1 = RRCNN_block(inc)
        self.down1 = DenseDownsample(inc+grate)
        self.conv2_1 = RRCNN_block(inc+2*grate)
        self.down2 = DenseDownsample(inc + 3*grate)
        self.conv3_1 = RRCNN_block(inc+4*grate)
        self.down3 = DenseDownsample(inc + 5*grate)
        self.conv4_1 = RRCNN_block(inc + 6*grate)
        self.down4 = DenseDownsample(inc + 7 * grate)
        self.conv_v = nn.Conv2d(inc+8*grate, 1024, 3, 1, 1)
        self.conv6_1 = RRCNN_dblock(inc+15*grate, 16 * base_chns, t=2)
        self.vit = ViT(img_dim=16,
                       in_channels=1024,
                       patch_dim=2,
                       dim=1024,
                       blocks=12,
                       heads=12,
                       dim_linear_block=3072,
                       classification=False)
        self.patches_back = nn.Linear(1024, 4096)
        self.vit_conv = nn.Sequential(
            nn.Conv2d(16 * base_chns, 32 * base_chns, (3, 3), padding=1),
            nn.PixelShuffle(2))
        self.conv7_1 = RRCNN_dblock(inc+9*grate, 8 * base_chns, t=2)
        self.conv8_1 = RRCNN_dblock(inc+5*grate, 4 * base_chns, t=2)
        self.conv9_1 = RRCNN_dblock(inc+2*grate, 2 * base_chns, t=2)
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64 * base_chns, 16 * base_chns, 4, 2, 1),
                                 )
        self.up6 = nn.Sequential(nn.ConvTranspose2d(16 * base_chns, 8 * base_chns, 4, 2, 1))
        self.up7 = nn.Sequential(nn.ConvTranspose2d(8 * base_chns, 4 * base_chns, 4, 2, 1))
        self.up8 = nn.Sequential(nn.ConvTranspose2d(4 * base_chns, 2 * base_chns, 4, 2, 1))
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * base_chns, out_channels=n_classes, kernel_size=3, padding=1),
        )
        self.CBAM1 = CBAM(inc+grate)
        self.CBAM2 = CBAM(inc+3*grate)
        self.CBAM3 = CBAM(inc+5*grate)
        self.CBAM4 = CBAM(inc+7*grate)
        self.CBAM5 = CBAM(1025)
        self.CBAM9 = CBAM(2 * base_chns, d=0)
        self.CBAM8 = CBAM(4 * base_chns, d=0)
        self.CBAM7 = CBAM(8 * base_chns, d=0)
        self.CBAM6 = CBAM(16 * base_chns, d=0)
        self.contour1 = Contour(inc+grate)
        self.contour2 = Contour(inc+3*grate)
        self.contour3 = Contour(inc+5*grate)
        self.contour4 = Contour(inc+7*grate)
        self.contour_up4 = Contour_up(inc+8*grate)
        self.contour_up3 = Contour_up(inc + 6 * grate)
        self.contour_up2 = Contour_up(inc + 4 * grate)
        self.contour_up1 = Contour_up(inc + 2 * grate)
        self.res1 = restruct(16 * base_chns)
        self.res2 = restruct(8 * base_chns)
        self.res3 = restruct(4 * base_chns)
        self.gn1 = gen_Block(16 * base_chns, 0)
        self.gn2 = gen_Block(8 * base_chns, 0)
        self.gn3 = gen_Block(4 * base_chns, 0)
        self.gn4 = gen_Block(2 * base_chns, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.CBAM1(self.conv1_1(x))
        down1 = self.down1(conv1)
        conv2 = self.CBAM2(self.conv2_1(down1))
        down2 = self.down2(conv2)
        conv3 = self.CBAM3(self.conv3_1(down2))
        down3 = self.down3(conv3)
        conv4 = self.CBAM4(self.conv4_1(down3))
        down4 = self.down4(conv4)

        conv_v = self.conv_v(down4)
        y = self.vit(conv_v)

        y = self.patches_back(y)
        y = rearrange(y, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=8, y=8, patch_x=2, patch_y=2)
        up5 = self.up5(y)
        conv4 = self.contour4(conv4+self.contour_up4(down4))
        out = t.cat((up5, conv4), 1)
        out = self.CBAM6(self.conv6_1(out))
        gn1 = self.gn1(out)
        up6 = self.up6(out)
        conv3 = self.contour3(conv3+self.contour_up3(down3))
        out = t.cat((up6, conv3), 1)
        out = self.CBAM7(self.conv7_1(out))
        gn2 = self.gn2(out)
        a64 = self.res1(gn1)
        gn_a = gn2 + self.res1(gn1)
        up7 = self.up7(out)
        conv2 = self.contour2(conv2+self.contour_up2(down2))
        out = t.cat((up7, conv2), 1)
        out = self.CBAM8(self.conv8_1(out))
        gn3 = self.gn3(out)
        a128 = self.res2(gn_a)
        gn_a = gn3 + self.res2(gn_a)
        up8 = self.up8(out)
        conv1 = self.contour1(conv1+self.contour_up1(down1))
        out = t.cat((up8, conv1), 1)
        out = self.CBAM9(self.conv9_1(out))
        gn4 = self.gn4(out)
        out = gn4 + self.res3(gn_a)

        return self.Th(out), self.Th(a128), self.Th(a64)


if __name__ == '__main__':
    x = t.randn((1, 3, 256, 256))
    seg = DRA_UNet()
    y, a, b = seg(x)
    print(y.shape)
    print(a.shape)
    print(b.shape)
