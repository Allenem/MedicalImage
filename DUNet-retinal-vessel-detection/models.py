import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        # return out
        return c10


class Unet_plus_plus(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_plus_plus, self).__init__()

        self.conv00 = DoubleConv(in_ch, 64)
        self.pool00 = nn.MaxPool2d(2)

        self.conv10 = DoubleConv(64, 128)
        self.pool10 = nn.MaxPool2d(2)
        self.up10 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv01 = DoubleConv(64*2, 64)

        self.conv20 = DoubleConv(128, 256)
        self.pool20 = nn.MaxPool2d(2)
        self.up20 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv11 = DoubleConv(128*2, 128)
        self.up11 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv02 = DoubleConv(64*3, 64)

        self.conv30 = DoubleConv(256, 512)
        self.pool30 = nn.MaxPool2d(2)
        self.up30 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv21 = DoubleConv(256*2, 256)
        self.up21 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv12 = DoubleConv(128*3, 128)
        self.up12 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv03 = DoubleConv(64*4, 64)

        self.conv40 = DoubleConv(512, 1024)
        self.up40 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv31 = DoubleConv(512*2, 512)
        self.up31 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv22 = DoubleConv(256*3, 256)
        self.up22 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv13 = DoubleConv(128*4, 128)
        self.up13 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv04 = DoubleConv(64*5, 64)

        self.conv_out = nn.Conv2d(64, out_ch, 1)
        self.conv_out_cat = nn.Conv2d(64*4, out_ch, 1)

    def forward(self, x):
        c00 = self.conv00(x)
        p00 = self.pool00(c00)

        c10 = self.conv10(p00)
        p10 = self.pool10(c10)
        up10 = self.up10(c10)
        merge00_10 = torch.cat([c00, up10], 1)
        c01 = self.conv01(merge00_10)
        cout01 = self.conv_out(c01)
        out01 = nn.Sigmoid()(cout01)

        c20 = self.conv20(p10)
        p20 = self.pool20(c20)
        up20 = self.up20(c20)
        merge10_20 = torch.cat([c10, up20], 1)
        c11 = self.conv11(merge10_20)
        up11 = self.up11(c11)
        merge00_01_11 = torch.cat([c00, c01, up11], 1)
        c02 = self.conv02(merge00_01_11)
        cout02 = self.conv_out(c02)
        out02 = nn.Sigmoid()(cout02)

        c30 = self.conv30(p20)
        p30 = self.pool30(c30)
        up30 = self.up30(c30)
        merge20_30 = torch.cat([c20, up30], 1)
        c21 = self.conv21(merge20_30)
        up21 = self.up21(c21)
        merge10_11_21 = torch.cat([c10, c11, up21], 1)
        c12 = self.conv12(merge10_11_21)
        up12 = self.up12(c12)
        merge00_01_02_12 = torch.cat([c00, c01, c02, up12], 1)
        c03 = self.conv03(merge00_01_02_12)
        cout03 = self.conv_out(c03)
        out03 = nn.Sigmoid()(cout03)

        c40 = self.conv40(p30)
        up40 = self.up40(c40)
        merge30_40 = torch.cat([c30, up40], 1)
        c31 = self.conv31(merge30_40)
        up31 = self.up31(c31)
        merge20_21_31 = torch.cat([c20, c21, up31], 1)
        c22 = self.conv22(merge20_21_31)
        up22 = self.up22(c22)
        merge10_11_12_22 = torch.cat([c10, c11, c12, up22], 1)
        c13 = self.conv13(merge10_11_12_22)
        up13 = self.up13(c13)
        merge00_01_02_03_13 = torch.cat([c00, c01, c02, c03, up13], 1)
        c04 = self.conv04(merge00_01_02_03_13)
        cout04 = self.conv_out(c04)
        out04 = nn.Sigmoid()(cout04)

        merge_c01_to_c04 = torch.cat([c01, c02, c03, c04], 1)
        cout_cat = self.conv_out_cat(merge_c01_to_c04)
        out_cat = nn.Sigmoid()(cout_cat)

        # return cout01, cout02, cout03, cout04, cout_cat
        # return out01, out02, out03, out04, out_cat
        return cout_cat
