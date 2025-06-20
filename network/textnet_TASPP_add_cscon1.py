# textnet0.py 表示原来公布、未修改的model

# 改进的model,对文本特征处理部分使用级联结构进行。
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.vgg import VggNet
from network.resnet import ResNet
from util.config import config as cfg
import time
from collections import OrderedDict



class UpBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class FeatureSelection(nn.Module):
    def __init__(self,
                 channel,
                 reduction=16):
        super(FeatureSelection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class horizontal_Dilate(nn.Module):  # in_channel 为各个小分支通道大小
    def __init__(self, in_channel, out_channel, output_stride=16):
        super(horizontal_Dilate, self).__init__()

        if output_stride == 16:
            dilations = [1, 2, 3]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        # self.fca1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channel, out_channel, 1, bias=False)),
        #                                           ('bn', nn.BatchNorm2d(out_channel)),
        #                                           ('relu', nn.ReLU(inplace=True))]))
        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(in_channel, in_channel, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(in_channel),
        #                                      nn.ReLU())
        self.fca2 = nn.Conv2d(in_channel, in_channel, kernel_size=(1,3), padding=(0,dilations[0]),
            dilation=dilations[0], bias=False)

        self.fca3 = nn.Conv2d(in_channel, in_channel, kernel_size=(1,5), padding=(0,2*dilations[1]),
            dilation=dilations[1], bias=False)

        self.fca4 = nn.Conv2d(in_channel, in_channel, kernel_size=(1,7), padding=(0,3*dilations[2]),
            dilation=dilations[2], bias=False)

        self.fca2_v = nn.Conv2d(in_channel, in_channel, kernel_size=(3,1), padding=(dilations[0], 0),
            dilation=dilations[0], bias=False)

        self.fca3_v = nn.Conv2d(in_channel, in_channel, kernel_size=(5,1), padding=(2*dilations[1], 0),
            dilation=dilations[1], bias=False)

        self.fca4_v = nn.Conv2d(in_channel, in_channel, kernel_size=(7,1), padding=(3*dilations[2], 0),
            dilation=dilations[2], bias=False)

        # 级联之后变为3 * in_channel输出
        self.ca = FeatureSelection(3 * in_channel)
        self.conv1 = nn.Conv2d(3 * in_channel, in_channel, 1, bias=False)

        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)



    def forward(self, x):
        x2 = self.fca2(x)
        # print("x2.shape is {}".format(x2.shape))
        x3 = self.fca3(x)
        # print("x3.shape is {}".format(x3.shape))
        # x3 = self._upsample(x3,x2)
        x4 = self.fca4(x)
        # print("x4.shape is {}".format(x4.shape))
        # x4 = self._upsample(x4, x2)


        # 针对上述特征输出进行正交方向的卷积处理
        x2_v = self.fca2_v(x2)
        # print("x2_v.shape is {}".format(x2_v.shape))

        x3_v = self.fca3_v(x3)
        # print("x3_v.shape is {}".format(x3_v.shape))
        # x3_v = self._upsample(x3_v, x2_v)

        x4_v = self.fca4_v(x4)
        # print("x4_v.shape is {}".format(x4_v.shape))
        # x4 = self._upsample(x4_v, x2_v)

        x5 = torch.cat((x2_v, x3_v, x4_v), dim=1)

        x5 = self.ca(x5)

        res = self.conv2(self.conv1(x5) + x)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.dropout(res)
        return res

    def _upsample(x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')



class RRGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FNUM = len(cfg.fuc_k)
        self.SepareConv0 = nn.Sequential(
            horizontal_Dilate(in_channels, in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        channels2 = in_channels + 1
        self.SepareConv1 = nn.Sequential(
            horizontal_Dilate(channels2, channels2),
            nn.Conv2d(channels2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        f_map = list()
        for i in range(self.FNUM):
            if i == 0:
                f = self.SepareConv0(x);
                f_map.append(f);
                continue
            b1 = torch.cat([x, f_map[i - 1]], dim=1)
            f = self.SepareConv1(b1)
            f_map.append(f)
        f_map = torch.cat(f_map, dim=1)
        return f_map


class FPN(nn.Module):

    def __init__(self, backbone='vgg_bn', pre_train=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "vgg" or backbone == 'vgg_bn':
            if backbone == 'vgg_bn':
                self.backbone = VggNet(name="vgg16_bn", pretrain=pre_train)
            elif backbone == 'vgg':
                self.backbone = VggNet(name="vgg16", pretrain=pre_train)

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            self.merge2 = UpBlok(128 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)

        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet101':
                self.backbone = ResNet(name="resnet101", pretrain=pre_train)
            elif backbone == 'resnet50':
                self.backbone = ResNet(name="resnet50", pretrain=pre_train)

            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 128)
            self.merge3 = UpBlok(512 + 128, 64)
            self.merge2 = UpBlok(256 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)
        else:
            print("backbone is not support !")

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)

        return up1, up2, up3, up4, up5


class TextNet(nn.Module):

    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, pre_train=is_training)
        self.rrgn = RRGN(16)

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x):
        end = time.time()
        up1, up2, up3, up4, up5 = self.fpn(x)
        b_time = time.time() - end
        end = time.time()
        predict_out = self.rrgn(up1)
        iter_time = time.time() - end

        return predict_out, b_time, iter_time
